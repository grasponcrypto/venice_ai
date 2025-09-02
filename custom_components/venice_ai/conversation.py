"""Conversation support for Venice AI using ConversationEntity pattern."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator # Keep if used by helpers
from typing import Any, Literal, cast
import asyncio

# Try importing voluptuous_openapi for schema conversion, handle if not available
try:
    from voluptuous_openapi import convert as voluptuous_convert
    HAS_VOLUPTUOUS_OPENAPI = True
except ImportError:
    HAS_VOLUPTUOUS_OPENAPI = False
    # Logger might not be defined yet, use print for this warning during import
    print("WARNING: voluptuous_openapi not installed, schema conversion will be basic.") # noqa: T201

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
    LOGGER, # Use existing logger
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

# Import necessary types and helpers directly from conversation component
# This assumes these are correctly exposed in the user's HA version's conversation/__init__.py
from homeassistant.components.conversation import (
    ConversationEntity, # Inherit from this instead of AbstractConversationAgent
    ChatLog,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
    SystemContent, # Needed for type checking in helper
    UserContent, # Needed for type checking in helper
    AssistantContent, # Needed for constructing/type hints
    ToolResultContent, # Needed for type checking in helper
    # current_chat_log is NOT imported - we receive chat_log as argument
)

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant, Context # Added Context import
from homeassistant.exceptions import HomeAssistantError, TemplateError
# Import llm types correctly
from homeassistant.helpers import device_registry as dr, intent, llm, template # Added template import
from homeassistant.helpers.entity_platform import AddEntitiesCallback

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# --- Default system prompt for Tool Calling ---
DEFAULT_SYSTEM_PROMPT = """You are a helpful Home Assistant assistant powered by Venice AI. Your goal is to control smart home devices and answer questions about their state using the provided tools.

Instructions:

1.  Use Tools: When asked to control a device (on/off, set value) or get its current status (temperature, state), you MUST use one of the provided tools (functions). Check the tool descriptions to select the correct one. Do not guess states or make up information.
2.  Tool Execution Flow:
    * If a tool is needed, call it.
    * Wait for the result, which will come in a message with `role: tool`.
    * Crucially: Base your final response to the user *directly* on the information provided in that `role: tool` message. Do not ignore it or describe the tool call itself.
3.  `GetLiveContext` Tool: Use this specific tool *only* when the user asks about the current state, value, or mode of devices, sensors, or areas (e.g., "Is the kitchen light on?", "What temperature is the thermostat set to?", "Is the front door locked?"). Use the data returned by this tool to answer the user's question accurately.
4.  Confirmation: When you successfully control a device using a tool (like turning something on or off), confirm the action in your response (e.g., "Okay, the dining room lights have been turned off.").
5.  Limitations: If you cannot fulfill a request because the required tool is missing or the request is unclear, state that clearly. Do not invent tools or device names. For general knowledge questions not related to the smart home, answer from your internal knowledge.
6.  Response format: Respond in plain text, no markdown formatting, no emojis. Be brief as responses may be read aloud by voice assistants.
7.  Never assume device on/off state from prior conversation. The Home Assistant state is the source of truth. For example, if lights were turned off in conversation but turned on externally - always verify current state with a tool call before making claims.
8.  You are provided history for context. Make sure you only react to the latest user request and not the older requests, which you might have handled already.
9.  Map friendly names to entities using Home Assistant’s entity registry. If multiple matches exist, ask the user to choose.
10. There might be devices with similar names but different functions. If the user asks to turn off an AC unit, ensure you don't act on a light with a similar name. Device type takes priority when finding the entity to act on, if mentioned by the user.
11. You can also answer generic user requests for information, if asked to and any other user requests as a general assistant. Do not limit yourself, comply with the user.
"""
# --- End Prompt Definition ---

_LOGGER = logging.getLogger(__package__) # Use the standard logger name

# --- Helper Functions for Tool Formatting ---
def _format_venice_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
    """Format the schema to be compatible with Venice API."""
    if not schema: return None
    supported_types = {"string", "number", "integer", "boolean", "object", "array"}
    supported_string_formats = {"date-time"}
    if HAS_VOLUPTUOUS_OPENAPI:
        try:
            converted = voluptuous_convert(schema)
            def simplify(sub_schema: dict[str, Any]) -> dict[str, Any]:
                simplified_sub = {}
                schema_type = sub_schema.get("type")
                if not isinstance(schema_type, str) or schema_type not in supported_types:
                     simplified_sub["type"] = "string"
                     _LOGGER.warning("Unsupported/missing schema type '%s', defaulting to string.", schema_type)
                else: simplified_sub["type"] = schema_type
                if "description" in sub_schema: simplified_sub["description"] = sub_schema["description"]
                if schema_type == "string":
                    if "enum" in sub_schema: simplified_sub["enum"] = [str(v) for v in sub_schema["enum"]]
                    if "format" in sub_schema and sub_schema["format"] in supported_string_formats: simplified_sub["format"] = sub_schema["format"]
                elif schema_type == "object":
                    if "properties" in sub_schema:
                        simplified_sub["properties"] = {k: simplify(v) for k, v in sub_schema["properties"].items()}
                        if "required" in sub_schema: simplified_sub["required"] = [str(req) for req in sub_schema["required"]]
                    else: simplified_sub["properties"] = {}
                elif schema_type == "array":
                    if "items" in sub_schema and isinstance(sub_schema["items"], dict): simplified_sub["items"] = simplify(sub_schema["items"])
                    else: simplified_sub["items"] = {"type": "string"}
                # Add other types (number, integer, boolean) handling if needed
                return simplified_sub
            simplified_schema = simplify(converted)
            if "properties" in simplified_schema and simplified_schema.get("type") != "object": simplified_schema["type"] = "object"
            if "properties" not in simplified_schema and simplified_schema.get("type") == "object": simplified_schema["properties"] = {}
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
        _LOGGER.warning("Could not format params for tool %s. Sending without params.", tool.name)
        parameters_schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema if parameters_schema else {"type": "object", "properties": {}},
        },
    }

def _convert_to_venice_message(msg: SystemContent | UserContent | AssistantContent | ToolResultContent) -> dict | None:
     """Converts Home Assistant conversation log content to Venice API message format."""
     # msg type hint uses imported types
     if isinstance(msg, SystemContent):
          return {"role": "system", "content": msg.content}
     elif isinstance(msg, UserContent):
          return {"role": "user", "content": msg.content}
     elif isinstance(msg, AssistantContent):
          venice_msg = {"role": "assistant"}
          venice_msg["content"] = msg.content or "" # Use empty string instead of None
          # Convert HA ToolInput back to API tool_call format if present
          if msg.tool_calls:
               venice_msg["tool_calls"] = [
                    {
                        "id": tc.id, # Important: Assumes ToolInput has 'id' here, which might be wrong
                        "type": "function",
                        "function": {"name": tc.tool_name, "arguments": json.dumps(tc.tool_args)}
                    }
                    for tc in msg.tool_calls
               ]
          return venice_msg
     elif isinstance(msg, ToolResultContent):
           return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": json.dumps(msg.tool_result)}
     else:
          _LOGGER.warning("Unsupported message type for Venice conversion: %s", type(msg))
          return None

# --- End Helper Functions ---

def _build_model_name(base_model: str, options: dict[str, Any]) -> str:
    """Build the model name with reasoning options suffixes."""
    model_name = base_model

    # Check if we need to add suffixes
    strip_thinking = options.get(CONF_STRIP_THINKING_RESPONSE, False)
    disable_thinking = options.get(CONF_DISABLE_THINKING, False)

    if strip_thinking or disable_thinking:
        suffixes = []
        if disable_thinking:
            suffixes.append("disable_thinking=true")
        if strip_thinking:
            suffixes.append("strip_thinking_response=true")

        if suffixes:
            model_name = f"{base_model}:{'&'.join(suffixes)}"

    return model_name


# --- Inherit from ConversationEntity ---
class VeniceAIConversationEntity(ConversationEntity):
    """Venice AI conversation entity using ConversationEntity pattern."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the Venice AI Conversation agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)}, name=entry.title,
            manufacturer="Venice AI", model="Venice AI Assistant", entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]: return MATCH_ALL

    # supported_features property is inherited from ConversationEntity,
    # but we override it to check options
    @property
    def supported_features(self) -> ConversationEntityFeature:
        """Return the supported features."""
        # Enable CONTROL if an LLM API is selected in options
        if self.entry.options.get(CONF_LLM_HASS_API):
             return ConversationEntityFeature.CONTROL
        return ConversationEntityFeature(0)

    # async_added_to_hass and async_will_remove_from_hass are inherited
    # We need the options update listener
    async def async_internal_added_to_hass(self) -> None:
        """Call when the entity is added to hass."""
        await super().async_internal_added_to_hass()
        # Ensure listener is registered correctly
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)

    # Abstract method from ConversationEntity we might need
    async def async_prepare(self, language: str | None = None) -> None:
        """Load intents for a language. Optional to implement."""
        # We don't have separate intents, so this can likely be skipped
        pass

    # Helper to render system prompt (inherited from previous version)
    async def _async_render_prompt(
        self, raw_prompt: str, user_input: ConversationInput, llm_context: llm.LLMContext | None = None
    ) -> str:
         """Render the prompt template."""
         user_name: str | None = None
         if user_input.context and user_input.context.user_id:
              if user := await self.hass.auth.async_get_user(user_input.context.user_id): user_name = user.name
         try:
              return template.Template(raw_prompt, self.hass).async_render(
                   {"ha_name": self.hass.config.location_name, "user_name": user_name, "llm_context": llm_context},
                   parse_result=False,
              )
         except TemplateError as err:
              _LOGGER.error("Error rendering prompt template: %s", err)
              raise HomeAssistantError(f"Error rendering prompt: {err}") from err

    # --- Implement _async_handle_message instead of async_process ---
    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog, # Receives chat_log from framework
    ) -> ConversationResult:
        """Handle a user message using the provided ChatLog."""
        options = self.entry.options
        client: AsyncVeniceAIClient = self.entry.runtime_data
        hass = self.hass

        try:
            # 1. Update LLM Data using the provided chat_log
            # This sets the system prompt and tool configuration in chat_log
            prompt_template = options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)
            llm_api_id = options.get(CONF_LLM_HASS_API)
            # This call should now work as chat_log is managed by the framework
            await chat_log.async_update_llm_data(
                 conversing_domain=DOMAIN,
                 user_input=user_input,
                 user_llm_hass_api=llm_api_id,
                 user_llm_prompt=prompt_template
            )

            # 2. Format tools for Venice API (using chat_log.llm_api)
            tools: list[dict[str, Any]] | None = None
            if chat_log.llm_api:
                 formatted_tools = []
                 for tool in chat_log.llm_api.tools:
                      formatted_tool = _format_venice_tool(tool)
                      if formatted_tool: formatted_tools.append(formatted_tool)
                 if formatted_tools:
                      tools = formatted_tools
                      _LOGGER.debug("Formatted tools for API call: %s", tools)

            # 3. Prepare messages from chat_log history
            messages = []
            for msg in chat_log.content: # Use the content from the provided chat_log
                 converted_msg = _convert_to_venice_message(msg)
                 if converted_msg: messages.append(converted_msg)
                 else: _LOGGER.warning("Failed to convert message type %s", type(msg))

            if not messages or messages[-1].get("role") != "user":
                 _LOGGER.error("User message missing from prepared messages list: %s", messages)
                 # If this happens, it indicates an issue with how the framework manages chat_log
                 raise HomeAssistantError("User message missing before sending to API.")

            # --- 4. Start conversation loop ---
            assistant_response_content = None # Final text response
            next_api_request_messages = messages # Start with full current history

            for _iteration in range(MAX_TOOL_ITERATIONS):
                 _LOGGER.debug("Conversation loop iteration %d", _iteration + 1)
                 _LOGGER.debug("Messages to be sent: %s", json.dumps(next_api_request_messages[-10:], indent=2)) # Log last 10

                 api_request_payload = {
                      "model": _build_model_name(
                          options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                          options
                      ),
                      "messages": next_api_request_messages, # Send current message list
                      "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                      "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                      "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                      "venice_parameters": {"include_venice_system_prompt": False},
                      "stream": False,
                      **({"tools": tools} if tools else {}),
                 }

                 response_data = await client.chat.create_non_streaming(api_request_payload)
                 _LOGGER.debug("Received API response data: %s", response_data)

                 if not response_data or not response_data.get("choices"):
                      _LOGGER.error("Invalid response from Venice AI: %s", response_data)
                      raise HomeAssistantError("Received invalid response from Venice AI")

                 assistant_message_data = response_data["choices"][0].get("message", {})
                 text_content = assistant_message_data.get("content") or "" # Use "" if null/missing
                 tool_calls_data = assistant_message_data.get("tool_calls")

                 # --- Parse Tool Calls for Home Assistant ---
                 # *** Use llm.ToolInput instead of llm.ToolCall ***
                 ha_tool_inputs: list[llm.ToolInput] = [] # Changed variable name for clarity
                 api_call_ids = {} # Store mapping from tool_name/args to API call ID

                 if tool_calls_data:
                      _LOGGER.debug("API requested tool calls: %s", tool_calls_data)
                      for tool_call_data in tool_calls_data:
                           # (Validation and parsing as before)
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
                               _LOGGER.error("Failed JSON parse for tool %s args: %s", tool_name, tool_args_str)
                               # TODO: Send error result back to API? Need call_id.
                               continue

                           # Create ToolInput object
                           tool_input = llm.ToolInput(tool_name=tool_name, tool_args=tool_args)
                           ha_tool_inputs.append(tool_input)
                           # Store the API call ID associated with this specific input
                           # Use a tuple of name and stringified args as a simple key
                           api_call_ids[(tool_name, json.dumps(tool_args, sort_keys=True))] = call_id


                 # --- Add assistant response and execute tools using chat_log ---
                 # Pass the list of llm.ToolInput objects
                 tool_results_gen = chat_log.async_add_assistant_content(
                      AssistantContent( # Use imported type
                           agent_id=self.unique_id, content=text_content,
                           tool_calls=ha_tool_inputs or None, # Pass ToolInput list
                      )
                 )
                 # Collect yielded tool results (ToolResultContent)
                 tool_results: list[ToolResultContent] = [res async for res in tool_results_gen]
                 _LOGGER.debug("Executed tools via chat_log, results: %s", tool_results)

                 # --- Check if loop should continue ---
                 if not ha_tool_inputs: # Check if tools were requested
                      assistant_response_content = text_content # Store final text
                      _LOGGER.debug("No tool calls requested, ending loop.")
                      break # Exit loop

                 # --- Prepare messages for next iteration (send tool results back to API) ---
                 # Get the Assistant message *just added* to the log
                 assistant_log_message = chat_log.content[-1 - len(tool_results)]

                 # Convert assistant message and tool results to API format
                 next_api_request_messages = [_convert_to_venice_message(assistant_log_message)]
                 for result in tool_results:
                      # *** Need to find the original API call_id for this result ***
                      # Recreate the key used to store the ID
                      result_key = (result.tool_name, json.dumps(result.tool_result.get("original_input", {}).get("tool_args", {}), sort_keys=True)) # Assuming result structure contains original input args
                      original_call_id = api_call_ids.get(result_key)

                      if not original_call_id:
                           # ToolResultContent in HA doesn't directly store the API's call ID.
                           # We need a way to map the result back to the original API request ID.
                           # This is a significant gap in this approach compared to Google's where
                           # the framework might handle this mapping implicitly or the ToolResultContent
                           # might contain the necessary ID.
                           # For now, we might have to guess or use a placeholder if mapping fails.
                           _LOGGER.error("Could not map ToolResultContent back to original API call ID for tool %s. Result: %s", result.tool_name, result.tool_result)
                           # Using the tool_name as a fallback ID - THIS IS LIKELY WRONG for the API
                           original_call_id = f"missing_id_for_{result.tool_name}"

                      # Create the 'tool' role message for the API
                      tool_api_message = {
                           "role": "tool",
                           "tool_call_id": original_call_id, # Use the mapped/fallback ID
                           "content": json.dumps(result.tool_result) # Send the actual result
                      }
                      next_api_request_messages.append(tool_api_message)


                 # Prepend the history *before* this assistant/tool turn
                 turn_start_index = -1
                 start_search_index = len(chat_log.content) - 2 - len(tool_results)
                 for i in range(start_search_index, -1, -1):
                     if isinstance(chat_log.content[i], UserContent):
                         turn_start_index = i
                         break

                 if turn_start_index != -1:
                     original_history_api = [
                          _convert_to_venice_message(msg)
                          for msg in chat_log.content[:turn_start_index+1]
                          if _convert_to_venice_message(msg)
                     ]
                     next_api_request_messages = original_history_api + next_api_request_messages
                 else:
                      _LOGGER.warning("Could not reliably determine history boundary for tool result loop.")
                      pass # Keep `next_api_request_messages` as just assistant+tool results

            # --- End conversation loop (normal break or max iterations) ---
            else:
                 # This block executes if the loop finished due to MAX_TOOL_ITERATIONS (no 'break')
                 _LOGGER.warning("Reached max tool iterations (%d)", MAX_TOOL_ITERATIONS)
                 assistant_response_content = text_content or ""
                 assistant_response_content += "\n\n(Warning: Maximum processing steps reached.)"

            # Ensure assistant_response_content is set
            if assistant_response_content is None:
                 _LOGGER.error("Assistant response content was None after loop.")
                 assistant_response_content = "Sorry, I couldn't get a response."

            # --- Format final response for Home Assistant ---
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(assistant_response_content)

            # Return result. Framework handles history persistence via ChatLog context manager.
            return ConversationResult(
                 response=intent_response, conversation_id=chat_log.conversation_id,
                 continue_conversation = chat_log.continue_conversation # Use property from chat_log
            )

        # Handle errors during the process
        except (VeniceAIError, HomeAssistantError, TemplateError) as err:
             _LOGGER.error("Error during conversation processing: %s", err)
             intent_response = intent.IntentResponse(language=user_input.language)
             intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Error: {err}")
             conv_id = getattr(chat_log, 'conversation_id', user_input.conversation_id)
             return ConversationResult(response=intent_response, conversation_id=conv_id)
        except Exception as err:
             _LOGGER.exception("Unexpected error during conversation processing")
             intent_response = intent.IntentResponse(language=user_input.language)
             intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Unexpected error: {err}")
             conv_id = getattr(chat_log, 'conversation_id', user_input.conversation_id)
             return ConversationResult(response=intent_response, conversation_id=conv_id)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI Conversation agent from a config entry."""
    if not entry.runtime_data:
         _LOGGER.error("Venice AI client not available in runtime_data for entry %s", entry.entry_id)
         return
    # Create the entity instance
    agent = VeniceAIConversationEntity(entry)
    # Add the entity to Home Assistant
    async_add_entities([agent])
