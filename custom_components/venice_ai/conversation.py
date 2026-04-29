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

# Import necessary types and helpers directly from conversation component
from homeassistant.components.conversation import (
    ConversationEntity,
    ChatLog,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
    SystemContent,
    UserContent,
    AssistantContent,
    ToolResultContent,
)

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import device_registry as dr, intent, llm, template
from homeassistant.helpers.entity_platform import AddEntitiesCallback

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

DEFAULT_SYSTEM_PROMPT = llm.DEFAULT_INSTRUCTIONS_PROMPT

_LOGGER = logging.getLogger(__package__)


def _make_schema_hashable(obj: Any) -> Any:
    """Recursively make schema objects hashable by replacing selectors with str."""
    if isinstance(obj, dict):
        return {k: _make_schema_hashable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_make_schema_hashable(item) for item in obj)
    elif hasattr(obj, "__class__") and "Selector" in obj.__class__.__name__:
        _LOGGER.debug(
            "_make_schema_hashable: replacing selector %s with str",
            obj.__class__.__name__,
        )
        return str
    else:
        return obj


def _format_venice_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
    """Format the schema to be compatible with Venice API."""
    if not schema:
        return None
    raw_schema = getattr(schema, "schema", None)
    if raw_schema is not None:
        schema = raw_schema
    _LOGGER.debug(
        "_format_venice_schema: schema type: %s, has schema attr: %s",
        type(schema),
        hasattr(schema, "schema"),
    )
    supported_types = {"string", "number", "integer", "boolean", "object", "array"}
    supported_string_formats = {"date-time"}
    if HAS_VOLUPTUOUS_OPENAPI:
        try:
            hashable_schema = _make_schema_hashable(schema)
            converted = voluptuous_convert(hashable_schema)

            def simplify(sub_schema: dict[str, Any]) -> dict[str, Any]:
                simplified_sub = {}
                if "anyOf" in sub_schema:
                    if sub_schema["anyOf"]:
                        return simplify(sub_schema["anyOf"][0])
                    else:
                        return {"type": "string"}
                schema_type = sub_schema.get("type")
                if not isinstance(schema_type, str) or schema_type not in supported_types:
                    simplified_sub["type"] = "string"
                    _LOGGER.warning(
                        "Unsupported/missing schema type '%s' in sub_schema %s, defaulting to string.",
                        schema_type,
                        sub_schema,
                    )
                else:
                    simplified_sub["type"] = schema_type
                if "description" in sub_schema:
                    simplified_sub["description"] = sub_schema["description"]
                if schema_type == "string":
                    if "enum" in sub_schema:
                        simplified_sub["enum"] = [str(v) for v in sub_schema["enum"]]
                    if (
                        "format" in sub_schema
                        and sub_schema["format"] in supported_string_formats
                    ):
                        simplified_sub["format"] = sub_schema["format"]
                elif schema_type == "object":
                    if "properties" in sub_schema:
                        simplified_sub["properties"] = {
                            k: simplify(v) for k, v in sub_schema["properties"].items()
                        }
                        if "required" in sub_schema:
                            simplified_sub["required"] = [str(req) for req in sub_schema["required"]]
                    else:
                        simplified_sub["properties"] = {}
                elif schema_type == "array":
                    if "items" in sub_schema and isinstance(sub_schema["items"], dict):
                        simplified_sub["items"] = simplify(sub_schema["items"])
                    else:
                        simplified_sub["items"] = {"type": "string"}
                return simplified_sub

            simplified_schema = simplify(converted)
            if "properties" in simplified_schema and simplified_schema.get("type") != "object":
                simplified_schema["type"] = "object"
            if "properties" not in simplified_schema and simplified_schema.get("type") == "object":
                simplified_schema["properties"] = {}
            return simplified_schema
        except Exception as e:
            _LOGGER.error("Failed to convert schema: %s", e, exc_info=True)
            return {"type": "object", "properties": {}}
    else:
        _LOGGER.warning("Cannot perform detailed schema conversion without voluptuous_openapi.")
        return {"type": "object", "properties": {}}


def _format_venice_tool(tool: llm.Tool) -> dict[str, Any] | None:
    """Format a Home Assistant Tool for the Venice API."""
    parameters_schema = _format_venice_schema(tool.parameters)
    if parameters_schema is None and tool.parameters:
        _LOGGER.warning(
            "Could not format params for tool %s. Sending without params.", tool.name
        )
        parameters_schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema if parameters_schema else {"type": "object", "properties": {}},
        },
    }


def _convert_to_venice_message(
    msg: SystemContent | UserContent | AssistantContent | ToolResultContent,
) -> dict | None:
    """Converts Home Assistant conversation log content to Venice API message format."""
    if isinstance(msg, SystemContent):
        return {"role": "system", "content": msg.content}
    elif isinstance(msg, UserContent):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, AssistantContent):
        venice_msg: dict[str, Any] = {"role": "assistant"}
        if msg.content and msg.content.strip():
            venice_msg["content"] = msg.content
        elif msg.tool_calls:
            venice_msg["content"] = "I'm processing your request..."
        else:
            venice_msg["content"] = msg.content or ""
        if msg.tool_calls:
            venice_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.tool_name, "arguments": json.dumps(tc.tool_args)},
                }
                for tc in msg.tool_calls
            ]
        return venice_msg
    elif isinstance(msg, ToolResultContent):
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": json.dumps(msg.tool_result),
        }
    else:
        _LOGGER.warning("Unsupported message type for Venice conversion: %s", type(msg))
        return None


def _build_model_name(base_model: str, options: dict[str, Any]) -> str:
    """Build the model name with reasoning options suffixes."""
    model_name = base_model
    strip_thinking = options.get(CONF_STRIP_THINKING_RESPONSE, False)
    disable_thinking = options.get(CONF_DISABLE_THINKING, False)

    if strip_thinking or disable_thinking:
        suffixes = []
        if disable_thinking:
            suffixes.append("disable_thinking=true")
        if strip_thinking:
            suffixes.append("strip_thinking_response=true")

        if suffixes:
            model_name = f"{base_model}:{ '&'.join(suffixes)}"

    return model_name


class VeniceAIConversationEntity(ConversationEntity):
    """Venice AI conversation entity using ConversationEntity pattern."""

    _attr_has_entity_name = True
    _attr_name = "VeniceAI Conversation"
    _attr_description = "Conversation agent"

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the Venice AI Conversation agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="Venice AI Assistant",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    @property
    def supported_features(self) -> ConversationEntityFeature:
        """Return the supported features."""
        if self.entry.options.get(CONF_LLM_HASS_API):
            return ConversationEntityFeature.CONTROL
        return ConversationEntityFeature(0)

    async def async_internal_added_to_hass(self) -> None:
        """Call when the entity is added to hass."""
        await super().async_internal_added_to_hass()
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)

    async def async_prepare(self, language: str | None = None) -> None:
        """Load intents for a language. Optional to implement."""
        pass

    async def _async_render_prompt(
        self,
        raw_prompt: str,
        user_input: ConversationInput,
        llm_context: llm.LLMContext | None = None,
    ) -> str:
        """Render the prompt template."""
        user_name: str | None = None
        if user_input.context and user_input.context.user_id:
            user = await self.hass.auth.async_get_user(user_input.context.user_id)
            if user:
                user_name = user.name
        try:
            return template.Template(raw_prompt, self.hass).async_render(
                {
                    "ha_name": self.hass.config.location_name,
                    "user_name": user_name,
                    "llm_context": llm_context,
                },
                parse_result=False,
            )
        except TemplateError as err:
            _LOGGER.error("Error rendering prompt template: %s", err)
            raise HomeAssistantError(f"Error rendering prompt: {err}") from err

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Handle a user message using the provided ChatLog."""
        options = self.entry.options
        client: AsyncVeniceAIClient = self.entry.runtime_data
        hass = self.hass

        try:
            # 1. Provide LLM data (prompt + selected tool providers) to the chat log
            prompt_template = (
                (options.get(CONF_PROMPT) or "")
                + ("\n\n" if options.get(CONF_PROMPT) else "")
                + DEFAULT_SYSTEM_PROMPT
            )
            llm_api_ids = options.get(CONF_LLM_HASS_API) or []
            if not isinstance(llm_api_ids, list):
                llm_api_ids = []
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                llm_api_ids,
                prompt_template,
                user_input.extra_system_prompt,
            )

            # 2. Format tools for Venice API
            tools: list[dict[str, Any]] | None = None
            if chat_log.llm_api:
                formatted_tools = []
                for tool in chat_log.llm_api.tools:
                    formatted_tool = _format_venice_tool(tool)
                    if formatted_tool:
                        formatted_tools.append(formatted_tool)
                if formatted_tools:
                    tools = formatted_tools

            # 3. Prepare messages from chat_log history (convert once, cache for loop)
            messages: list[dict[str, Any]] = []
            for msg in chat_log.content:
                converted_msg = _convert_to_venice_message(msg)
                if converted_msg:
                    messages.append(converted_msg)
                else:
                    _LOGGER.warning("Failed to convert message type %s", type(msg))

            if not messages or messages[-1].get("role") != "user":
                _LOGGER.error("User message missing from prepared messages list: %s", messages)
                raise HomeAssistantError("User message missing before sending to API.")

            assistant_response_content = None

            # --- 4. Start conversation loop ---
            for _iteration in range(MAX_TOOL_ITERATIONS):
                api_request_payload = {
                    "model": _build_model_name(
                        options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                        options,
                    ),
                    "messages": messages,
                    "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    "venice_parameters": {"include_venice_system_prompt": False},
                    "stream": False,
                    **({"tools": tools} if tools else {}),
                }

                response_data = await client.chat.create_non_streaming(api_request_payload)

                if not response_data or not response_data.get("choices"):
                    _LOGGER.error("Invalid response from Venice AI: %s", response_data)
                    raise HomeAssistantError("Received invalid response from Venice AI")

                assistant_message_data = response_data["choices"][0].get("message", {})
                text_content = assistant_message_data.get("content") or ""
                tool_calls_data = assistant_message_data.get("tool_calls")

                # --- Parse Tool Calls for Home Assistant ---
                ha_tool_inputs: list[llm.ToolInput] = []
                if tool_calls_data:
                    for tool_call_data in tool_calls_data:
                        call_id = tool_call_data.get("id")
                        call_type = tool_call_data.get("type")
                        func_details = tool_call_data.get("function")
                        if not call_id or call_type != "function" or not func_details:
                            _LOGGER.warning("Skipping malformed tool call: %s", tool_call_data)
                            continue
                        tool_name = func_details.get("name")
                        tool_args_str = func_details.get("arguments", "{}")
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

                        tool_input = llm.ToolInput(
                            tool_name=tool_name, tool_args=tool_args, id=call_id
                        )
                        ha_tool_inputs.append(tool_input)

                # --- Add assistant response and execute tools using chat_log ---
                tool_results_gen = chat_log.async_add_assistant_content(
                    AssistantContent(
                        agent_id=self.unique_id,
                        content=text_content,
                        tool_calls=ha_tool_inputs or None,
                    )
                )
                tool_results: list[ToolResultContent] = [
                    res async for res in tool_results_gen
                ]

                # Convert assistant message and append to cached messages
                assistant_msg = _convert_to_venice_message(
                    chat_log.content[-1 - len(tool_results)]
                )
                if assistant_msg:
                    messages.append(assistant_msg)

                # --- Check if loop should continue ---
                if not ha_tool_inputs:
                    assistant_response_content = text_content
                    break

                # --- Append tool results to cached messages ---
                for result in tool_results:
                    tool_api_message = {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": json.dumps(result.tool_result),
                    }
                    messages.append(tool_api_message)

            # --- End conversation loop ---
            else:
                _LOGGER.warning("Reached max tool iterations (%d)", MAX_TOOL_ITERATIONS)
                assistant_response_content = text_content or ""
                assistant_response_content += "\n\n(Warning: Maximum processing steps reached.)"

            if assistant_response_content is None:
                _LOGGER.error("Assistant response content was None after loop.")
                assistant_response_content = "Sorry, I couldn't get a response."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(assistant_response_content)

            return ConversationResult(
                response=intent_response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=chat_log.continue_conversation,
            )

        except (VeniceAIError, HomeAssistantError, TemplateError) as err:
            _LOGGER.error("Error during conversation processing: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN, f"Error: {err}"
            )
            conv_id = getattr(chat_log, "conversation_id", user_input.conversation_id)
            return ConversationResult(response=intent_response, conversation_id=conv_id)
        except Exception as err:
            _LOGGER.exception("Unexpected error during conversation processing")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN, f"Unexpected error: {err}"
            )
            conv_id = getattr(chat_log, "conversation_id", user_input.conversation_id)
            return ConversationResult(response=intent_response, conversation_id=conv_id)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI Conversation agent from a config entry."""
    if not entry.runtime_data:
        _LOGGER.error(
            "Venice AI client not available in runtime_data for entry %s",
            entry.entry_id,
        )
        return
    agent = VeniceAIConversationEntity(entry)
    async_add_entities([agent])
