"""Conversation support for Venice AI using ConversationEntity pattern."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

# Try importing voluptuous_openapi for schema conversion, handle if not available
try:
    from voluptuous_openapi import convert as voluptuous_convert
    HAS_VOLUPTUOUS_OPENAPI = True
except ImportError:
    HAS_VOLUPTUOUS_OPENAPI = False

# Import client exceptions and client itself
from .client import AsyncVeniceAIClient, VeniceAIError
# Import constants for default values and keys
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_STRIP_THINKING_RESPONSE,
    CONF_DISABLE_THINKING,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

_LOGGER = logging.getLogger(__name__)

from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
    ChatLog,
    UserContent,
    AssistantContent,
    SystemContent,
    ToolResultContent,
    ConverseError,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import intent, llm, device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.template import Template
from homeassistant.util import ulid as ulid_util

# Default system prompt for Venice AI
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant controlling a smart home. You can control lights, switches, climate, media players, and other devices. Always be concise and helpful."""

# Maximum number of tool iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 5


def _make_schema_hashable(obj: Any) -> Any:
    """Recursively convert a voluptuous schema into a hashable representation."""
    if isinstance(obj, dict):
        return frozenset((k, _make_schema_hashable(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return tuple(_make_schema_hashable(v) for v in obj)
    if hasattr(obj, "__class__") and "Selector" in obj.__class__.__name__:
        _LOGGER.debug(
            "_make_schema_hashable: replacing selector %s with str",
            obj.__class__.__name__,
        )
        return str
    return obj


def _format_venice_schema(raw_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a voluptuous dict schema into a Venice-compatible OpenAPI-like schema."""
    schema = {}
    for key, val in raw_schema.items():
        if val is str:
            schema[key] = {"type": "string"}
        elif val is int:
            schema[key] = {"type": "integer"}
        elif val is float:
            schema[key] = {"type": "number"}
        elif val is bool:
            schema[key] = {"type": "boolean"}
        elif isinstance(val, dict):
            schema[key] = {"type": "object", "properties": _format_venice_schema(val)}
        elif isinstance(val, list):
            schema[key] = {
                "type": "array",
                "items": _format_venice_schema({"__item__": val[0]}).get("__item__", {}),
            }
        elif val is Any:
            schema[key] = {"type": "string"}
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
            hashable = _make_schema_hashable(tool.parameters)
            # Use voluptuous_openapi to convert
            parameters_schema = voluptuous_convert(hashable)
            # voluptuous_openapi may return list for 'anyOf' patterns; simplify
            if isinstance(parameters_schema, list):
                parameters_schema = parameters_schema[0]
            if isinstance(parameters_schema, dict) and "properties" in parameters_schema:
                # Recursively ensure all sub-schemas have a type
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
                # Wrap simple dict in properties
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
        _LOGGER.warning("Cannot perform detailed schema conversion without voluptuous_openapi.")
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
            # Handle thinking tags if needed
            content = msg.content
            if strip_thinking and "<think>" in content:
                parts = content.split("</think>")
                if len(parts) > 1:
                    content = parts[-1].strip()
            messages.append({"role": "assistant", "content": content})
        elif isinstance(msg, ToolResultContent):
            # Venice expects tool results as function results
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


class VeniceAIConversationEntity(ConversationEntity):
    """Venice AI conversation entity."""

    def __init__(self, entry: ConfigEntry, client: AsyncVeniceAIClient) -> None:
        """Initialize the entity."""
        self.entry = entry
        self._client = client
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        self._attr_name = entry.title
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="Venice AI Conversation",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return ["en"]

    @property
    def supported_options(self) -> list[str]:
        """Return list of supported options."""
        return [CONF_PROMPT, CONF_CHAT_MODEL, CONF_MAX_TOKENS, CONF_TEMPERATURE, CONF_TOP_P]

    async def async_process(
        self, user_input: ConversationInput, context: Any = None
    ) -> ConversationResult:
        """Process a conversation input."""
        options = self.entry.options
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        max_tokens = options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
        temperature = options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
        top_p = options.get(CONF_TOP_P, RECOMMENDED_TOP_P)
        strip_thinking = options.get(CONF_STRIP_THINKING_RESPONSE, False)
        disable_thinking = options.get(CONF_DISABLE_THINKING, False)
        prompt_template_str = options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)
        llm_api = options.get(CONF_LLM_HASS_API)

        # Render system prompt template
        try:
            prompt_template = Template(prompt_template_str)
            system_prompt = prompt_template.render()
        except TemplateError as err:
            _LOGGER.error("Error rendering prompt template: %s", err)
            raise HomeAssistantError(f"Error rendering prompt: {err}") from err

        # Set up LLM API if configured
        tools: list[llm.Tool] = []
        if llm_api:
            try:
                api = await llm.async_get_api(self.hass, llm_api)
                tools = api.tools
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

        # Create chat log
        chat_log = ChatLog(
            conversation_id=user_input.conversation_id or ulid_util.ulid_now(),
            content=[UserContent(content=user_input.text)],
        )

        assistant_response_content = None
        text_content = ""

        try:
            for iteration in range(MAX_TOOL_ITERATIONS):
                messages = _convert_chat_log_to_venice_messages(
                    chat_log, system_prompt, strip_thinking=strip_thinking
                )

                if not messages or messages[-1].get("role") != "user":
                    _LOGGER.error("User message missing from prepared messages list: %s", messages)
                    raise HomeAssistantError("User message missing before sending to API.")

                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=venice_tools if venice_tools else None,
                    stream=False,
                )

                response_data = response
                if not response_data or not response_data.get("choices"):
                    _LOGGER.error("Invalid response from Venice AI: %s", response_data)
                    raise HomeAssistantError("Received invalid response from Venice AI")

                choice = response_data["choices"][0]
                message = choice.get("message", {})
                text_content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    assistant_response_content = text_content
                    break

                # Process tool calls
                assistant_content = AssistantContent(
                    agent_id="venice_ai",
                    content=text_content,
                )
                chat_log.content.append(assistant_content)

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

                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        _LOGGER.error(
                            "Failed JSON parse for tool %s args: %s", tool_name, tool_args_str
                        )
                        continue

                    # Find matching tool and invoke
                    tool_result = None
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = await tool.target(self.hass, tool_args, user_input.context)
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
            else:
                _LOGGER.warning("Reached max tool iterations (%d)", MAX_TOOL_ITERATIONS)
                assistant_response_content = text_content or ""

            if assistant_response_content is None:
                _LOGGER.error("Assistant response content was None after loop.")
                assistant_response_content = "Sorry, I couldn't get a response."

        except (VeniceAIError, HomeAssistantError, TemplateError) as err:
            _LOGGER.error("Error during conversation processing: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Error: {err}",
            )
            return ConversationResult(
                conversation_id=chat_log.conversation_id,
                response=intent_response,
            )
        except Exception as err:
            _LOGGER.exception("Unexpected error during conversation processing")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Unexpected error occurred.",
            )
            return ConversationResult(
                conversation_id=chat_log.conversation_id,
                response=intent_response,
            )

        # Build response
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(assistant_response_content)

        return ConversationResult(
            conversation_id=chat_log.conversation_id,
            response=intent_response,
        )

    @callback
    def async_internal_added_to_hass(self) -> None:
        """Register update listener."""
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_updated)
        )

    @callback
    def _async_entry_updated(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle options update."""
        self.entry = entry


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI conversation entity."""
    if not entry.runtime_data:
        _LOGGER.error(
            "Venice AI client not available in runtime_data for entry %s",
            entry.entry_id,
        )
        return

    entity = VeniceAIConversationEntity(
        entry,
        entry.runtime_data,
    )
    async_add_entities([entity])
