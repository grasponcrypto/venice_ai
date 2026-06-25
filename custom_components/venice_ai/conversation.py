"""Conversation support for Venice AI using ConversationEntity pattern."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from homeassistant.components.conversation import (
    HOME_ASSISTANT_AGENT,
    MATCH_ALL,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
    ChatLog,
    UserContent,
    AssistantContent,
    SystemContent,
    ToolResultContent,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import intent, llm, device_registry as dr, selector
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.template import Template
from homeassistant.util import ulid as ulid_util

from .client import RateLimitError, VeniceAIError
from .venice_api import ChatParameters, VeniceConversationService
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_STRIP_THINKING_RESPONSE,
    CONF_DISABLE_THINKING,
    RECOMMENDED_DISABLE_THINKING,
    CONF_STREAM_RESPONSE,
    RECOMMENDED_STREAM_RESPONSE,
    CONVERSATION_TTL_SECONDS,
    DOMAIN,
    HAS_VOLUPTUOUS_OPENAPI,
    MAX_CHAT_HISTORY_SIZE,
    MAX_CHAT_LOG_LENGTH,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

if HAS_VOLUPTUOUS_OPENAPI:
    from voluptuous_openapi import convert as voluptuous_convert  # type: ignore[import-untyped]

_LOGGER = logging.getLogger(__name__)

# Default system prompt for Venice AI
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant controlling a smart home. You can control lights, switches, climate, media players, and other devices. Always be concise and helpful."""

def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> (or  thinking… end of thinking ) blocks from model output.

    Handles both the XML-style tags used by some reasoning models and the
    literal `` thinking`` / `` end of thinking`` markers emitted by Venice AI.
    """
    if not text:
        return text
    # XML-style <think>…</think>
    while True:
        start = text.lower().find("<think>")
        if start == -1:
            break
        end = text.lower().find("</think>", start)
        if end == -1:
            # unmatched open tag – strip to end to be safe
            text = text[:start].strip()
            break
        text = text[:start] + text[end + 8:]
    # Venice-style  thinking… end of thinking
    if " thinking" in text:
        parts = text.split(" end of thinking")
        if len(parts) > 1:
            text = parts[-1].strip()
    return text.strip()


def _convert_schema_to_hashable(obj: Any) -> Any:
    """Recursively convert a voluptuous schema into a hashable representation."""
    if isinstance(obj, dict):
        return frozenset((k, _convert_schema_to_hashable(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return tuple(_convert_schema_to_hashable(v) for v in obj)
    if isinstance(obj, selector.Selector):
        _LOGGER.debug(
            "_convert_schema_to_hashable: replacing selector %s with str",
            obj.__class__.__name__,
        )
        return str
    return obj


def _format_venice_schema(raw_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a schema dict into a Venice-compatible OpenAPI-like schema.

    Recursively traverses the schema, preserving nested dict/list structures
    and selector metadata (e.g., SelectSelector options become JSON Schema
    ``enum`` values) while mapping Python types to JSON Schema types.
    """
    schema: dict[str, Any] = {}
    for key, val in raw_schema.items():
        # Unwrap voluptuous Required/Optional wrappers
        if hasattr(val, "schema"):
            val = val.schema

        if val is str or val is Any:
            schema[key] = {"type": "string"}
        elif val is int:
            schema[key] = {"type": "integer"}
        elif val is float:
            schema[key] = {"type": "number"}
        elif val is bool:
            schema[key] = {"type": "boolean"}
        elif isinstance(val, selector.SelectSelector):
            config = getattr(val, "config", None)
            options = getattr(config, "options", None) if config else None
            if options:
                schema[key] = {"type": "string", "enum": list(options)}
                _LOGGER.debug(
                    "_format_venice_schema: preserved SelectSelector with %d options for key %s",
                    len(options),
                    key,
                )
            else:
                schema[key] = {"type": "string"}
        elif isinstance(val, selector.Selector):
            _LOGGER.debug(
                "_format_venice_schema: converting selector %s to string for key %s",
                val.__class__.__name__,
                key,
            )
            schema[key] = {"type": "string"}
        elif isinstance(val, dict):
            schema[key] = {"type": "object", "properties": _format_venice_schema(val)}
        elif isinstance(val, list):
            schema[key] = {
                "type": "array",
                "items": _format_venice_schema({"__item__": val[0]}).get("__item__", {}),
            }
        else:
            _LOGGER.debug(
                "_format_venice_schema: unsupported type %s for key %s, defaulting to string",
                type(val).__name__,
                key,
            )
            schema[key] = {"type": "string"}
    return schema


def _convert_tool_parameters(tool: llm.Tool) -> dict[str, Any] | None:
    """Convert tool parameters schema to Venice-compatible format."""
    if not tool.parameters:
        return None

    if HAS_VOLUPTUOUS_OPENAPI:
        try:
            hashable = _convert_schema_to_hashable(tool.parameters)
            parameters_schema = voluptuous_convert(hashable)
            # voluptuous_openapi may return list for 'anyOf' patterns; simplify
            if isinstance(parameters_schema, list):
                parameters_schema = parameters_schema[0]
            if isinstance(parameters_schema, dict) and "properties" in parameters_schema:
                def _ensure_types(sub_schema: dict[str, Any]) -> None:
                    if "properties" in sub_schema:
                        for _, prop in sub_schema["properties"].items():
                            if isinstance(prop, dict):
                                if "type" not in prop:
                                    if "properties" in prop:
                                        prop["type"] = "object"
                                    elif "enum" in prop:
                                        prop["type"] = "string"
                                    else:
                                        prop["type"] = "string"
                                _ensure_types(prop)

                _ensure_types(parameters_schema)
                return parameters_schema
            elif isinstance(parameters_schema, dict):
                return {"type": "object", "properties": parameters_schema}
            else:
                _LOGGER.warning(
                    "Unexpected schema type from voluptuous_openapi: %s",
                    type(parameters_schema).__name__,
                )
                return {"type": "object", "properties": {}}
        except Exception as e:
            _LOGGER.error("Failed to convert schema: %s", e, exc_info=True)
            return {"type": "object", "properties": {}}
    else:
        _LOGGER.debug("Cannot perform detailed schema conversion without voluptuous_openapi.")
        return {"type": "object", "properties": {}}


def _convert_chat_log_to_venice_messages(
    chat_log: ChatLog,
    system_prompt: str,
    strip_thinking: bool = False,
) -> list[dict[str, Any]]:
    """Convert Home Assistant ChatLog to Venice AI message format."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in chat_log.content:
        if isinstance(msg, UserContent):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AssistantContent):
            raw = _strip_thinking(msg.content) if strip_thinking else msg.content
            # Try to decode JSON-encoded assistant messages that carry
            # tool_calls metadata embedded during async_process.
            tool_calls = None
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, dict) and "text" in decoded:
                    raw = decoded["text"]
                    tool_calls = decoded.get("tool_calls")
            except (json.JSONDecodeError, TypeError):
                tool_calls = None
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": raw}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
        elif isinstance(msg, ToolResultContent):
            messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": json.dumps(msg.tool_result),
            })
        elif isinstance(msg, SystemContent):
            messages.append({"role": "system", "content": msg.content})
        else:
            _LOGGER.warning("Unsupported message type for Venice conversion: %s", type(msg))

    return messages


def _trim_chat_log(chat_log: ChatLog) -> None:
    """Trim chat log to prevent unbounded growth during long conversations.

    Preserves the first user message and the most recent messages up to
    MAX_CHAT_LOG_LENGTH. System and tool-result messages are trimmed first.
    """
    content = chat_log.content
    if len(content) <= MAX_CHAT_LOG_LENGTH:
        return

    keep_first = [content[0]]
    tail = content[-(MAX_CHAT_LOG_LENGTH - 1):]
    trimmed = keep_first + tail
    _LOGGER.debug(
        "Trimmed chat log from %d to %d messages", len(content), len(trimmed)
    )
    chat_log.content.clear()
    chat_log.content.extend(trimmed)


class VeniceAIConversationEntity(ConversationEntity):
    """Venice AI conversation entity."""

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self._client = entry.runtime_data.client
        # ARCH-1/ARCH-2: delegate API interaction to the service layer.
        self._service = VeniceConversationService(self._client)
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        self._attr_name = entry.title
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="Conversation",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        # Conversation history keyed by conversation_id.
        # OrderedDict used as an LRU cache: entries are moved to the tail on
        # each access and the oldest (head) entry is evicted when the dict
        # exceeds MAX_CHAT_HISTORY_SIZE, preventing unbounded memory growth.
        self._chat_logs: OrderedDict[str, ChatLog] = OrderedDict()
        # MED-2: cache compiled Template objects keyed by their source string so
        # the parse step is skipped on subsequent turns. The string is used as
        # the cache key because Template objects themselves are mutable and may
        # not be hashable. Stored on the entity so it survives across turns.
        self._template_cache: dict[str, Template] = {}
        # HIGH-2: monotonic timestamps recording the most-recent access time of
        # each conversation. Read by the periodic cleanup loop so the
        # conversation cache can be trimmed when it grows.
        self._last_access: dict[str, float] = {}
        # HIGH-2: background task that periodically evicts idle conversations.
        # The task is started in async_added_to_hass and cancelled on unload.
        self._cleanup_task: asyncio.Task[None] | None = None
        # LOW-3: last conversation id we responded to; surfaced via
        # extra_state_attributes so automations can detect recent activity.
        self._last_conversation_id: str | None = None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """LOW-3: surface the live conversation state for the entity card.

        Returning at least ``active_conversations`` (count of in-memory chat
        logs) is non-sensitive and lets users/automations reason about
        resource usage. ``last_conversation_id`` and ``model`` are exposed
        for transparency.
        """
        return {
            "active_conversations": len(self._chat_logs),
            "last_conversation_id": self._last_conversation_id,
            "model": self.entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
        }

    async def _periodic_cleanup(self) -> None:
        """HIGH-2: background loop that evicts idle conversations.

        Sleeps for ``CONVERSATION_TTL_SECONDS`` between sweeps. Errors are
        logged and the loop continues rather than terminating, so a transient
        failure during cleanup does not stop future sweeps.
        """
        while True:
            try:
                await asyncio.sleep(CONVERSATION_TTL_SECONDS)
                self._cleanup_old_conversations(time.monotonic())
            except asyncio.CancelledError:
                # Normal shutdown path
                raise
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Error during periodic conversation cleanup")

    def _cleanup_old_conversations(self, now: float) -> None:
        """HIGH-2: evict conversations idle for more than CONVERSATION_TTL_SECONDS.

        When the cache is at or above MAX_CHAT_HISTORY_SIZE, evict the
        oldest half of conversations (LRU order) so the next request has
        room to allocate. Conversations with no recorded ``_last_access``
        are treated as never-used (newly created) and skipped.
        """
        cutoff = now - CONVERSATION_TTL_SECONDS
        # Single pass: gather candidates and evict in one shot.
        evicted = 0
        # First, evict anything strictly older than the TTL.
        for cid in list(self._last_access.keys()):
            if self._last_access.get(cid, 0.0) < cutoff:
                self._chat_logs.pop(cid, None)
                self._last_access.pop(cid, None)
                evicted += 1
        # If still over the LRU limit, evict the oldest half by OrderedDict order.
        if len(self._chat_logs) > MAX_CHAT_HISTORY_SIZE:
            target = len(self._chat_logs) - (MAX_CHAT_HISTORY_SIZE // 2 or 1)
            while evicted < target and self._chat_logs:
                oldest_id, _ = self._chat_logs.popitem(last=False)
                self._last_access.pop(oldest_id, None)
                evicted += 1
        if evicted:
            _LOGGER.debug(
                "HIGH-2 periodic cleanup evicted %d conversation(s); %d remain",
                evicted,
                len(self._chat_logs),
            )

    def _cancel_cleanup(self) -> None:
        """HIGH-2: cancel the periodic cleanup task on unload."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        self._cleanup_task = None

    def _get_or_create_chat_log(self, conversation_id: str | None) -> ChatLog:
        """Return the persisted ChatLog for *conversation_id*, or create a new one.

        Uses the OrderedDict as an LRU cache:
        - On hit: move the entry to the tail (most-recently-used position).
        - On miss: create a new empty ChatLog, insert at the tail, and evict the
          oldest entry (head) if the cache exceeds MAX_CHAT_HISTORY_SIZE.

        The caller is responsible for appending the new user message after this
        method returns.
        """
        cid = conversation_id or ulid_util.ulid_now()
        if cid in self._chat_logs:
            # Promote to most-recently-used
            self._chat_logs.move_to_end(cid)
            # HIGH-2: record access time so the periodic cleanup loop can
            # distinguish active vs idle conversations.
            self._last_access[cid] = time.monotonic()
            _LOGGER.debug("Resuming existing conversation %s (%d messages)", cid, len(self._chat_logs[cid].content))
            return self._chat_logs[cid]

        # New conversation
        chat_log = ChatLog(self.hass, conversation_id=cid, content=[])
        self._chat_logs[cid] = chat_log
        self._last_access[cid] = time.monotonic()
        # Evict least-recently-used if over the limit
        if len(self._chat_logs) > MAX_CHAT_HISTORY_SIZE:
            evicted_id, _ = self._chat_logs.popitem(last=False)
            self._last_access.pop(evicted_id, None)
            _LOGGER.debug(
                "Evicted oldest conversation %s from history cache (limit=%d)",
                evicted_id,
                MAX_CHAT_HISTORY_SIZE,
            )
        _LOGGER.debug("Started new conversation %s", cid)
        return chat_log

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return MATCH_ALL

    @property
    def supported_options(self) -> list[str]:
        """Return list of supported options."""
        return [CONF_PROMPT, CONF_CHAT_MODEL, CONF_MAX_TOKENS, CONF_TEMPERATURE, CONF_TOP_P, CONF_MAX_TOOL_ITERATIONS, CONF_STRIP_THINKING_RESPONSE, CONF_DISABLE_THINKING, CONF_STREAM_RESPONSE]

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a conversation input."""
        options = self.entry.options
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        max_tokens = options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
        temperature = options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
        top_p = options.get(CONF_TOP_P, RECOMMENDED_TOP_P)
        strip_thinking = options.get(CONF_STRIP_THINKING_RESPONSE, False)
        prompt_template_str = options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)
        llm_api = options.get(CONF_LLM_HASS_API)

        # Render system prompt template with Home Assistant context so
        # template functions (e.g. now(), states(), area_entities()) work.
        # MED-2: cache compiled Template objects keyed by their source string so
        # repeated turns with the same prompt skip the parse step.
        try:
            prompt_template = self._template_cache.get(prompt_template_str)
            if prompt_template is None:
                prompt_template = Template(prompt_template_str, self.hass)
                # Bound the cache so a user editing the prompt to many distinct
                # strings cannot grow the cache unboundedly.
                if len(self._template_cache) >= 32:
                    self._template_cache.pop(next(iter(self._template_cache)))
                self._template_cache[prompt_template_str] = prompt_template
            system_prompt = prompt_template.async_render()
        except TemplateError as err:
            _LOGGER.error("Error rendering prompt template: %s", err)
            raise HomeAssistantError(f"Error rendering prompt: {err}") from err

        # Set up LLM API if configured
        tools: list[llm.Tool] = []
        if llm_api:
            try:
                # LLMContext signature changed across HA versions:
                # - HA < 2024.7: accepts user_prompt, language, assistant, device_id
                # - HA ≥ 2024.7: may drop some kwargs or reorder them
                # Try the full signature first; fall back to a minimal one.
                try:
                    llm_context = llm.LLMContext(
                        platform=DOMAIN,
                        context=user_input.context,
                        user_prompt=user_input.text,
                        language=user_input.language,
                        assistant=HOME_ASSISTANT_AGENT,
                        device_id=user_input.device_id,
                    )
                except TypeError:
                    # Newer HA dropped some kwargs — use positional/minimal form
                    llm_context = llm.LLMContext(
                        platform=DOMAIN,
                        context=user_input.context,
                        language=user_input.language,
                        assistant=HOME_ASSISTANT_AGENT,
                        device_id=user_input.device_id,
                    )
                api = await llm.async_get_api(self.hass, llm_api, llm_context)
                tools = list(api.tools)
            except Exception as err:
                _LOGGER.warning("Failed to get LLM API %s: %s", llm_api, err)

        # Convert tools to Venice format
        venice_tools = []
        for tool in tools:
            tool_dict: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                },
            }
            parameters_schema = _convert_tool_parameters(tool)
            if parameters_schema is None and tool.parameters:
                _LOGGER.warning(
                    "Could not format params for tool %s. Sending without params.", tool.name
                )
            else:
                tool_dict["function"]["parameters"] = parameters_schema or {"type": "object", "properties": {}}
            venice_tools.append(tool_dict)

        # NumberSelector stores values as floats; cast to int so range() works.
        max_tool_iterations = int(options.get(CONF_MAX_TOOL_ITERATIONS, RECOMMENDED_MAX_TOOL_ITERATIONS))

        _LOGGER.debug(
            "Conversation turn starting: llm_api=%r, tools_available=%d, model=%s, "
            "max_tool_iterations=%d, stream=%s",
            llm_api,
            len(venice_tools),
            model,
            max_tool_iterations,
            options.get(CONF_STREAM_RESPONSE, RECOMMENDED_STREAM_RESPONSE),
        )
        if venice_tools:
            _LOGGER.debug(
                "Tools being sent to API: %s",
                [t["function"]["name"] for t in venice_tools],
            )

        # Retrieve existing chat log or create a new one, then append the new user message.
        # History is persisted across calls so the model has full multi-turn context.
        chat_log = self._get_or_create_chat_log(user_input.conversation_id)
        chat_log.content.append(UserContent(content=user_input.text))

        assistant_response_content = None
        text_content = ""

        try:
            _trim_chat_log(chat_log)
            _turn_start = time.monotonic()
            _total_prompt_tokens = 0
            _total_completion_tokens = 0
            _LOGGER.debug(
                "[PERF] [+0.000s] Turn received at %s — conversation=%s, user_text=%d chars, "
                "model=%s, stream=%s, tools=%d",
                datetime.datetime.now().isoformat(timespec="milliseconds"),
                chat_log.conversation_id,
                len(user_input.text),
                model,
                options.get(CONF_STREAM_RESPONSE, RECOMMENDED_STREAM_RESPONSE),
                len(venice_tools),
            )

            for iteration in range(max_tool_iterations):
                messages = _convert_chat_log_to_venice_messages(
                    chat_log, system_prompt, strip_thinking=strip_thinking
                )

                if not messages:
                    _LOGGER.error("Message list is empty before sending to API.")
                    raise HomeAssistantError("Message list is empty before sending to API.")

                disable_thinking = options.get(CONF_DISABLE_THINKING, RECOMMENDED_DISABLE_THINKING)
                venice_params: dict[str, Any] | None = None
                if disable_thinking:
                    venice_params = {"disable_thinking": True}

                # ARCH-1/ARCH-2 + MED-3: delegate the API interaction to the
                # service layer. When streaming is enabled the service consumes
                # the streaming chat API and reassembles deltas (and tool-call
                # fragments) into a response shaped like the non-streaming one,
                # so the downstream parsing logic is identical for both paths.
                stream_response = options.get(
                    CONF_STREAM_RESPONSE, RECOMMENDED_STREAM_RESPONSE
                )
                chat_params = ChatParameters(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=venice_tools if venice_tools else None,
                    venice_parameters=venice_params,
                )

                _elapsed_so_far = time.monotonic() - _turn_start
                if iteration == 0:
                    _call_label = "Initial API call"
                else:
                    _call_label = f"API call after tool results (iteration {iteration + 1})"
                _LOGGER.debug(
                    "[PERF] [+%.3fs] %s → Venice AI: messages=%d, tools=%d, stream=%s",
                    _elapsed_so_far,
                    _call_label,
                    len(messages),
                    len(venice_tools),
                    stream_response,
                )
                _iter_start = time.monotonic()

                if stream_response:
                    stream_result = await self._service.chat_stream(
                        messages, chat_params
                    )
                    response_data = stream_result.as_response()
                else:
                    stream_result = None
                    response_data = await self._service.chat(messages, chat_params)

                _iter_elapsed = time.monotonic() - _iter_start

                if not isinstance(response_data, dict):
                    _LOGGER.error(
                        "Invalid response type from Venice AI: %s",
                        type(response_data).__name__,
                    )
                    raise HomeAssistantError(
                        f"Received invalid response type: {type(response_data).__name__}"
                    )

                if not response_data.get("choices"):
                    _LOGGER.error("Invalid response from Venice AI: %s", response_data)
                    raise HomeAssistantError("Received invalid response")

                # Log token usage if available in the response
                _elapsed_after = time.monotonic() - _turn_start
                usage = response_data.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    _total_prompt_tokens += prompt_tokens
                    _total_completion_tokens += completion_tokens
                    _LOGGER.debug(
                        "[PERF] [+%.3fs] Response from Venice AI (%s): %.3fs, "
                        "tokens prompt=%d completion=%d total=%d%s",
                        _elapsed_after,
                        _call_label,
                        _iter_elapsed,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        f", first_token=+{stream_result.time_to_first_token:.3f}s"
                        if stream_result and stream_result.time_to_first_token is not None
                        else "",
                    )
                else:
                    _LOGGER.debug(
                        "[PERF] [+%.3fs] Response from Venice AI (%s): %.3fs%s",
                        _elapsed_after,
                        _call_label,
                        _iter_elapsed,
                        f", first_token=+{stream_result.time_to_first_token:.3f}s"
                        if stream_result and stream_result.time_to_first_token is not None
                        else "",
                    )

                choice = response_data["choices"][0]
                if not isinstance(choice, dict):
                    _LOGGER.error(
                        "Invalid choice type in Venice AI response: %s",
                        type(choice).__name__,
                    )
                    raise HomeAssistantError("Received invalid choice format")

                finish_reason = choice.get("finish_reason", "unknown")
                _LOGGER.debug(
                    "[PERF] [+%.3fs] API call #%d finish_reason=%r",
                    time.monotonic() - _turn_start,
                    iteration + 1,
                    finish_reason,
                )

                # Warn explicitly when the model was cut off by the token limit.
                # This is the most common cause of empty or truncated responses.
                if finish_reason == "length":
                    _LOGGER.warning(
                        "[PERF] API call #%d stopped due to max_tokens limit (%d). "
                        "The response was TRUNCATED — increase Max Tokens in the integration options "
                        "if responses are incomplete. Total elapsed: +%.3fs",
                        iteration + 1,
                        max_tokens,
                        time.monotonic() - _turn_start,
                    )

                message = choice.get("message", {})
                if not isinstance(message, dict):
                    _LOGGER.error(
                        "Invalid message type in Venice AI response: %s",
                        type(message).__name__,
                    )
                    raise HomeAssistantError("Received invalid message format")

                raw_text_content = message.get("content", "") or ""
                if strip_thinking:
                    text_content = _strip_thinking(raw_text_content)
                    # Always log before/after so we can diagnose reasoning models
                    # that emit only <think> blocks (which strip to nothing).
                    _LOGGER.debug(
                        "[PERF] [+%.3fs] strip_thinking applied: raw=%d chars → stripped=%d chars "
                        "| RAW (first 500): %r | STRIPPED (first 500): %r",
                        time.monotonic() - _turn_start,
                        len(raw_text_content),
                        len(text_content),
                        raw_text_content[:500],
                        text_content[:500],
                    )
                else:
                    text_content = raw_text_content
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    _total_elapsed = time.monotonic() - _turn_start
                    visible_content = text_content.strip()

                    if not visible_content:
                        # Empty response — determine why and surface a helpful message.
                        if finish_reason == "length":
                            # Model hit the max_tokens cap before producing any output.
                            _LOGGER.warning(
                                "[PERF] [+%.3fs] finish_reason=length AND empty content — "
                                "response was fully truncated (max_tokens=%d)",
                                _total_elapsed,
                                max_tokens,
                            )
                            assistant_response_content = (
                                "I'm sorry, my response was cut off because it exceeded the maximum "
                                "token limit. Please try a shorter question or increase the Max Tokens "
                                "setting in the Venice AI integration options."
                            )
                        elif strip_thinking:
                            # Most likely the model returned ONLY a <think> block and no
                            # visible text after it — common with reasoning models when
                            # strip_thinking is enabled. Return the raw (un-stripped) text
                            # so the user gets something rather than silence.
                            raw_content = message.get("content", "")
                            if raw_content.strip():
                                _LOGGER.warning(
                                    "[PERF] [+%.3fs] strip_thinking removed ALL content "
                                    "(finish_reason=%r) — returning raw model output so user "
                                    "sees a response. Consider disabling strip_thinking or "
                                    "using a model that emits text outside <think> blocks. "
                                    "Raw content (first 200 chars): %r",
                                    _total_elapsed,
                                    finish_reason,
                                    raw_content[:200],
                                )
                                assistant_response_content = raw_content
                            else:
                                _LOGGER.warning(
                                    "[PERF] [+%.3fs] Model returned empty content "
                                    "(finish_reason=%r, strip_thinking=True) — "
                                    "raw content is also empty",
                                    _total_elapsed,
                                    finish_reason,
                                )
                                assistant_response_content = (
                                    "I didn't receive a response from the model. "
                                    "Please try again."
                                )
                        else:
                            _LOGGER.warning(
                                "[PERF] [+%.3fs] Model returned empty content "
                                "(finish_reason=%r) — returning fallback message",
                                _total_elapsed,
                                finish_reason,
                            )
                            assistant_response_content = (
                                "I didn't receive a response from the model. "
                                "Please try again."
                            )
                    else:
                        _LOGGER.debug(
                            "[PERF] [+%.3fs] No tool calls — final response ready after %d API call(s). "
                            "Total tokens this turn: prompt=%d, completion=%d (%.3fs total)",
                            _total_elapsed,
                            iteration + 1,
                            _total_prompt_tokens,
                            _total_completion_tokens,
                            _total_elapsed,
                        )
                        assistant_response_content = text_content
                    break

                # Store assistant message with tool call metadata encoded so
                # _convert_chat_log_to_venice_messages can reconstruct the
                # full assistant+tool_calls payload the API expects.
                encoded_content = json.dumps({
                    "text": text_content,
                    "tool_calls": tool_calls,
                })
                assistant_content = AssistantContent(
                    agent_id=DOMAIN,
                    content=encoded_content,
                )
                chat_log.content.append(assistant_content)

                _LOGGER.debug(
                    "[PERF] [+%.3fs] Tool call(s) requested (%d): %s — dispatching now",
                    time.monotonic() - _turn_start,
                    len(tool_calls),
                    [tc.get("function", {}).get("name") for tc in tool_calls],
                )

                for tool_call_data in tool_calls:
                    call_id = tool_call_data.get("id")
                    func_details = tool_call_data.get("function", {})
                    call_type = tool_call_data.get("type", "function")
                    tool_name = func_details.get("name")
                    tool_args_str = func_details.get("arguments", "{}")

                    if not call_id or call_type != "function" or not func_details:
                        _LOGGER.warning("Skipping malformed tool call: %s", tool_call_data)
                        continue
                    if not tool_name:
                        _LOGGER.warning("Tool call missing name: %s", tool_call_data)
                        continue

                    _LOGGER.debug(
                        "[PERF] [+%.3fs] Calling HA tool: %s, args: %s",
                        time.monotonic() - _turn_start,
                        tool_name,
                        tool_args_str,
                    )

                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        _LOGGER.error(
                            "Failed JSON parse for tool %s args: %s", tool_name, tool_args_str
                        )
                        continue

                    # SEC-2: validate that tool args are a JSON object before
                    # invoking the tool. A non-object (e.g. array, string, or
                    # number) almost certainly indicates a malformed or hostile
                    # model output and must not be passed to HA tools that
                    # expect keyword arguments.
                    if not isinstance(tool_args, dict):
                        _LOGGER.warning(
                            "SEC-2: tool %s returned non-object args (%s); skipping",
                            tool_name,
                            type(tool_args).__name__,
                        )
                        tool_result = {
                            "error": f"Tool {tool_name} arguments must be a JSON object",
                        }
                        chat_log.content.append(
                            ToolResultContent(
                                tool_call_id=call_id,
                                tool_result=tool_result,
                            )
                        )
                        continue

                    # Find matching tool and invoke via the public HA LLM API
                    tool_result = None
                    _tool_start = time.monotonic()
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                tool_input = llm.ToolInput(
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    platform=DOMAIN,
                                    context=user_input.context,
                                    user_prompt=user_input.text,
                                    assistant=HOME_ASSISTANT_AGENT,
                                    device_id=user_input.device_id,
                                )
                                tool_result = await tool.async_call(self.hass, tool_input)
                                _LOGGER.debug(
                                    "[PERF] [+%.3fs] HA tool %s returned in %.3fs: %s",
                                    time.monotonic() - _turn_start,
                                    tool_name,
                                    time.monotonic() - _tool_start,
                                    tool_result,
                                )
                            except Exception as tool_err:
                                _LOGGER.warning("Tool %s failed: %s", tool_name, tool_err)
                                tool_result = {"error": str(tool_err)}
                            break

                    if tool_result is None:
                        _LOGGER.warning("Tool %s not found", tool_name)
                        tool_result = {"error": f"Tool {tool_name} not found"}

                    tool_result_content = ToolResultContent(
                        tool_call_id=call_id,
                        tool_result=tool_result,
                    )
                    chat_log.content.append(tool_result_content)

                # Trim chat log to prevent unbounded growth after processing all tool calls
                _trim_chat_log(chat_log)
            else:
                _LOGGER.warning(
                    "Reached max tool iterations (%d). Total tokens this turn: "
                    "prompt=%d, completion=%d (%.2fs total)",
                    max_tool_iterations,
                    _total_prompt_tokens,
                    _total_completion_tokens,
                    time.monotonic() - _turn_start,
                )
                assistant_response_content = text_content or ""

            if assistant_response_content is None:
                _LOGGER.error("Assistant response content was None after loop.")
                assistant_response_content = "Sorry, I couldn't get a response."

        except RateLimitError as err:
            # QUAL-1: route rate-limit through the same error path as other
            # recoverable errors so the user-facing message is consistent and
            # the conversation_id is still returned (lets the caller resume).
            _LOGGER.warning("Rate limit hit during conversation: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Venice AI rate limit exceeded. Please wait a moment and try again.",
            )
            return ConversationResult(
                conversation_id=chat_log.conversation_id,
                response=intent_response,
            )
        except (VeniceAIError, HomeAssistantError, TemplateError) as err:
            _LOGGER.error("Error during conversation processing: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Venice AI error: {err}",
            )
            return ConversationResult(
                conversation_id=chat_log.conversation_id,
                response=intent_response,
            )
        except Exception:
            # QUAL-1: never leak internal exception details to end users; the
            # logger.exception call above already captured the traceback.
            _LOGGER.exception("Unexpected error during conversation processing")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "An unexpected error occurred while contacting Venice AI. See logs for details.",
            )
            return ConversationResult(
                conversation_id=chat_log.conversation_id,
                response=intent_response,
            )

        # Persist the final assistant turn so subsequent calls see the full history.
        chat_log.content.append(
            AssistantContent(agent_id=DOMAIN, content=assistant_response_content)
        )
        _trim_chat_log(chat_log)

        # Build response. LOW-3: surface the active conversation id in the
        # extra_state_attributes property via _last_conversation_id so the
        # entity card and automations can see recent activity.
        self._last_conversation_id = chat_log.conversation_id
        intent_response = intent.IntentResponse(language=user_input.language)
        _total_turn_elapsed = time.monotonic() - _turn_start
        intent_response.async_set_speech(assistant_response_content)
        _LOGGER.debug(
            "[PERF] [+%.3fs] Response dispatched to user — conversation=%s, "
            "%d messages in log, response=%d chars: %r",
            _total_turn_elapsed,
            chat_log.conversation_id,
            len(chat_log.content),
            len(assistant_response_content),
            assistant_response_content[:200] if assistant_response_content else "<empty>",
        )

        return ConversationResult(
            conversation_id=chat_log.conversation_id,
            response=intent_response,
        )

    async def async_added_to_hass(self) -> None:
        """Start the HIGH-2 cleanup loop.

        NOTE: No add_update_listener is registered here because VeniceAIOptionsFlow
        subclasses OptionsFlowWithReload (HA ≥ 2024.1), which automatically reloads
        the entire integration when options are saved. Registering an update listener
        alongside OptionsFlowWithReload raises:
            ValueError: Config entry update listeners should not be used with OptionsFlowWithReload
        The entity is recreated fresh on every reload, so a manual listener is unnecessary.
        """
        # HIGH-2: start the periodic cleanup task. Cancelled on unload.
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = self.hass.async_create_background_task(
                self._periodic_cleanup(), name="venice_ai_conversation_cleanup"
            )
            self.entry.async_on_unload(self._cancel_cleanup)

    async def async_will_remove_from_hass(self) -> None:
        """HIGH-2: stop the cleanup task when the entity is being removed."""
        self._cancel_cleanup()


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI conversation entity."""
    from . import VeniceAIRuntimeData

    runtime_data: VeniceAIRuntimeData = entry.runtime_data
    if not runtime_data or not runtime_data.client:
        _LOGGER.error(
            "Venice AI client not available in runtime_data for entry %s",
            entry.entry_id,
        )
        return

    entity = VeniceAIConversationEntity(entry)
    async_add_entities([entity])
