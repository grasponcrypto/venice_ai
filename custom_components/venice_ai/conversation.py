"""Conversation support for Venice AI."""

from collections.abc import AsyncGenerator, Callable, Iterable
import json
import re
import uuid
from typing import Any, Literal, cast

from .client import AsyncVeniceAIClient, VeniceAIError, ChatCompletionChunk
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.util import dt as dt_util
import logging

MAX_TOOL_ITERATIONS = 10

_LOGGER = logging.getLogger(__name__)

def _format_message_content(
    entities: list[tuple[str, str | None]],
    services: dict[str, list[str]],
    prompt: str,
    api_prompt: str,
    tools: list[str],
) -> str:
    """Format system message content with entities, states, services, and tools."""
    entities_text = []
    for entity_id, state in entities:
        state_str = state if state is not None else "unknown"
        entities_text.append(f"- {entity_id} (state: {state_str})")
    entities_text_str = "\n".join(entities_text) if entities_text else "None"

    services_text = []
    for entity_id, service_list in services.items():
        for service in service_list:
            if any(sensitive in service.lower() for sensitive in ['lock', 'alarm', 'security']):
                continue
            services_text.append(f"- {service} (for {entity_id})")
    services_text_str = "\n".join(services_text) if services_text else "None"

    tools_text = []
    for tool in tools:
        tools_text.append(f"- {tool}")
    tools_text_str = "\n".join(tools_text) if tools_text else "None"

    return (
        f"{api_prompt}\n\n"
        f"You are a Home Assistant controller. For commands like 'turn on the office light' or 'open the garage door', "
        f"use the provided tools (e.g., HassTurnOn, HassTurnOff) to execute actions. Always include a JSON tool call "
        f"and a text confirmation (e.g., 'The Office light is now on') in your response, using this format:\n"
        f"```json\n"
        f"{{\n"
        f"  \"id\": \"<unique_id>\",\n"
        f"  \"function\": {{\n"
        f"    \"name\": \"<tool_name>\",\n"
        f"    \"arguments\": {{\"name\": \"<entity_name>\", \"domain\": \"<entity_domain>\"}}\n"
        f"  }}\n"
        f"}}\n"
        f"```\n"
        f"Examples:\n"
        f"- For 'turn on the office light':\n"
        f"Text: 'The Office light is now on'\n"
        f"```json\n"
        f"{{\n"
        f"  \"id\": \"call_1234\",\n"
        f"  \"function\": {{\n"
        f"    \"name\": \"HassTurnOn\",\n"
        f"    \"arguments\": {{\"name\": \"Office\", \"domain\": \"light\"}}\n"
        f"  }}\n"
        f"}}\n"
        f"```\n"
        f"- For 'turn off the kitchen fan':\n"
        f"Text: 'The Kitchen fan is now off'\n"
        f"```json\n"
        f"{{\n"
        f"  \"id\": \"call_5678\",\n"
        f"  \"function\": {{\n"
        f"    \"name\": \"HassTurnOff\",\n"
        f"    \"arguments\": {{\"name\": \"Kitchen\", \"domain\": \"fan\"}}\n"
        f"  }}\n"
        f"}}\n"
        f"```\n"
        f"If no action is required (e.g., status query), respond with text only, no tool call.\n"
        f"User-defined prompt (if any):\n{prompt}\n\n"
        f"Available Home Assistant entities and their current states:\n{entities_text_str}\n\n"
        f"Available Home Assistant services for these entities:\n{services_text_str}\n\n"
        f"Available tools:\n{tools_text_str}"
    )

def _parse_api_prompt(api_prompt: str) -> list[str]:
    """Parse api_prompt to extract entity IDs."""
    entities = []
    pattern = r"- names: (.+?)\n\s+domain: (\w+)"
    matches = re.findall(pattern, api_prompt, re.MULTILINE)
    for name, domain in matches:
        clean_name = name.strip("'").strip()
        entity_id = f"{domain}.{clean_name.lower().replace(' ', '_').replace(':', '_').replace('[', '').replace(']', '').replace(',', '').replace('(', '').replace(')', '').replace('-', '_').replace('\'', '')}"
        entities.append(entity_id)
    _LOGGER.debug("Parsed entities from api_prompt: %s", entities)
    return entities

def _get_controllable_domains() -> set[str]:
    """Return domains that can be controlled by intent tools."""
    return {"light", "fan", "switch", "cover", "climate", "media_player", "vacuum", "scene", "button", "select"}

def _is_control_command(user_input: str) -> bool:
    """Determine if the user input is a control command."""
    control_keywords = r"(?:set|turn|switch|adjust|open|close|dim|brighten)\s+(?:to\s+)?(?:on|off|\d+%?)"
    if re.search(control_keywords, user_input, re.IGNORECASE):
        _LOGGER.debug("Input detected as control command: %s", user_input)
        return True
    question_keywords = r"\b(is|are|what|where|when|how|can|does|do)\b"
    if re.search(question_keywords, user_input, re.IGNORECASE):
        _LOGGER.debug("Input detected as question: %s", user_input)
        return False
    _LOGGER.debug("Input not clearly a control command or question, assuming non-control: %s", user_input)
    return False

def _parse_response_for_tool_call(content: str, user_input: str, entities: list[tuple[str, str | None]]) -> list[dict[str, Any]]:
    """Parse natural language response for tool call intents."""
    tool_calls = []
    entity_names = [entity_id.split(".")[1].replace("_", " ").title() for entity_id, _ in entities]
    _LOGGER.debug("Available entity names for validation: %s", entity_names)

    # Skip tool call parsing for non-control commands (e.g., questions)
    if not _is_control_command(user_input):
        _LOGGER.debug("Skipping tool call parsing for non-control input: %s", user_input)
        return tool_calls

    # Pattern for explicit tool call statements in response
    pattern = r"I will use the (\w+) tool with name ['\"]([^'\"]+)['\"] and domain ['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, content, re.IGNORECASE)
    for tool_name, name, domain in matches:
        if tool_name in ["HassTurnOn", "HassTurnOff", "HassLightSet", "HassSetPosition", "HassClimateSetTemperature"] and name in entity_names:
            tool_calls.append({
                "id": f"call_{uuid.uuid4()}",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps({"name": name, "domain": domain}),
                },
            })

    # Fallback: Parse user input for control commands
    if not tool_calls:
        fallback_pattern = r"(?:turn\s+(on|off)\s+(?:the\s+)?([\w\s]+?)(?:\s+(light|fan|switch|cover|climate|media\splayer|vacuum|scene|button|select))?\b)"
        input_match = re.search(fallback_pattern, user_input, re.IGNORECASE)
        if input_match:
            action, name, domain = input_match.groups()
            name = name.strip().title()
            domain = domain.lower() if domain else "light"  # Default to light if unclear
            
            # Validate entity name
            if name not in entity_names:
                _LOGGER.warning("Parsed entity name '%s' not in available entities: %s", name, entity_names)
                return tool_calls
            
            tool_name = "HassTurnOn" if action.lower() == "on" else "HassTurnOff"
            tool_calls.append({
                "id": f"call_{uuid.uuid4()}",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps({"name": name, "domain": domain}),
                },
            })

    _LOGGER.debug("Parsed tool calls from response: %s", tool_calls)
    return tool_calls

def _get_tool_schema(tools: list[str]) -> list[dict[str, Any]]:
    """Generate tool schema for Venice AI API."""
    schema = []
    for tool in tools:
        if tool in ["HassTurnOn", "HassTurnOff"]:
            schema.append({
                "type": "function",
                "function": {
                    "name": tool,
                    "description": f"Execute the {tool} action in Home Assistant",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the entity"},
                            "domain": {"type": "string", "description": "Domain of the entity (e.g., light, fan)"},
                        },
                        "required": ["name", "domain"],
                    },
                },
            })
    return schema

def _convert_content(
    chat_content: Iterable[conversation.Content],
    hass: HomeAssistant,
) -> list[dict[str, Any]]:
    """Transform HA chat_log content into Venice AI API format."""
    messages: list[dict[str, Any]] = []
    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            messages.append({
                "role": "tool",
                "tool_call_id": content.tool_call_id,
                "content": json.dumps(content.tool_result),
            })
        elif isinstance(content, conversation.UserContent):
            if not messages or messages[-1]["role"] != "user":
                messages.append({
                    "role": "user",
                    "content": content.content,
                })
            else:
                messages[-1]["content"] += f"\n{content.content}"
        elif isinstance(content, conversation.AssistantContent):
            if not messages or messages[-1]["role"] != "assistant":
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [],
                })
            if content.content:
                messages[-1]["content"] += content.content
            if content.tool_calls:
                messages[-1]["tool_calls"].extend([
                    {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args),
                        },
                    }
                    for tool_call in content.tool_calls
                ])
        elif isinstance(content, conversation.SystemContent):
            pass
        else:
            raise TypeError(f"Unexpected content type: {type(content)}")
    return messages

async def _transform_stream(
    result: AsyncGenerator[ChatCompletionChunk, None],
    messages: list[dict[str, Any]],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the Venice AI response stream into HA format."""
    full_response = ""
    tool_calls: dict[str, dict[str, Any]] = {}  # Track tool calls by ID
    index_to_id: dict[int, str] = {}  # Map index to tool call ID

    async for chunk in result:
        _LOGGER.debug("Received response chunk: %s", chunk)
        choice = chunk.choices[0] if chunk.choices else {}
        delta = choice.get("delta", {})
        message = choice.get("message", {})

        if "content" in delta and delta["content"] is not None:
            full_response += delta["content"]
            yield {"content": delta["content"]}
        elif "content" in message and message["content"] is not None:
            full_response += message["content"]
            yield {"content": message["content"]}

        if "tool_calls" in delta and delta["tool_calls"]:
            for tool_call in delta["tool_calls"]:
                if not isinstance(tool_call, dict):
                    _LOGGER.warning("Malformed tool call in delta: %s", tool_call)
                    continue

                tool_id = tool_call.get("id")
                tool_index = tool_call.get("index", 0)

                if tool_id:
                    if tool_id not in tool_calls:
                        tool_calls[tool_id] = {"id": tool_id, "function": {}}
                        index_to_id[tool_index] = tool_id
                    _LOGGER.debug("Associated tool call ID %s with index %d", tool_id, tool_index)
                elif tool_index in index_to_id:
                    tool_id = index_to_id[tool_index]
                    _LOGGER.debug("Matched tool call with index %d to ID %s", tool_index, tool_id)
                else:
                    _LOGGER.warning("Tool call missing ID and unmatched index %d: %s", tool_index, tool_call)
                    continue

                if "function" in tool_call:
                    function_data = tool_call["function"]
                    if isinstance(function_data, dict):
                        if "name" in function_data:
                            tool_calls[tool_id]["function"]["name"] = function_data["name"]
                            _LOGGER.debug("Set name for tool call %s: %s", tool_id, function_data["name"])
                        if "arguments" in function_data and function_data["arguments"]:
                            if "arguments" not in tool_calls[tool_id]["function"]:
                                tool_calls[tool_id]["function"]["arguments"] = ""
                            tool_calls[tool_id]["function"]["arguments"] += function_data["arguments"]
                            _LOGGER.debug("Appended arguments for tool call %s: %s", tool_id, function_data["arguments"])

                # Check if tool call is complete
                if (
                    tool_id in tool_calls
                    and "name" in tool_calls[tool_id]["function"]
                    and "arguments" in tool_calls[tool_id]["function"]
                    and tool_calls[tool_id]["function"]["arguments"]
                ):
                    try:
                        tool_args = json.loads(tool_calls[tool_id]["function"]["arguments"])
                        _LOGGER.debug("Complete tool call for ID %s: %s", tool_id, tool_calls[tool_id])
                        yield {
                            "tool_calls": [
                                llm.ToolInput(
                                    id=tool_id,
                                    tool_name=tool_calls[tool_id]["function"]["name"],
                                    tool_args=tool_args,
                                )
                            ]
                        }
                        # Clear the tool call after yielding
                        del tool_calls[tool_id]
                        del index_to_id[tool_index]
                    except json.JSONDecodeError as err:
                        _LOGGER.warning("Invalid tool call arguments for ID %s: %s, error: %s", tool_id, tool_calls[tool_id]["function"]["arguments"], err)
                else:
                    _LOGGER.debug("Incomplete tool call for ID %s: %s", tool_id, tool_calls[tool_id])

        elif "tool_calls" in message and message["tool_calls"]:
            for tool_call in message["tool_calls"]:
                if not isinstance(tool_call, dict):
                    _LOGGER.warning("Malformed tool call in message: %s", tool_call)
                    continue

                tool_id = tool_call.get("id")
                tool_index = tool_call.get("index", 0)

                if tool_id:
                    if tool_id not in tool_calls:
                        tool_calls[tool_id] = {"id": tool_id, "function": {}}
                        index_to_id[tool_index] = tool_id
                    _LOGGER.debug("Associated tool call ID %s with index %d", tool_id, tool_index)
                elif tool_index in index_to_id:
                    tool_id = index_to_id[tool_index]
                    _LOGGER.debug("Matched tool call with index %d to ID %s", tool_index, tool_id)
                else:
                    _LOGGER.warning("Tool call missing ID and unmatched index %d: %s", tool_index, tool_call)
                    continue

                if "function" in tool_call:
                    function_data = tool_call["function"]
                    if isinstance(function_data, dict):
                        if "name" in function_data:
                            tool_calls[tool_id]["function"]["name"] = function_data["name"]
                            _LOGGER.debug("Set name for tool call %s: %s", tool_id, function_data["name"])
                        if "arguments" in function_data and function_data["arguments"]:
                            if "arguments" not in tool_calls[tool_id]["function"]:
                                tool_calls[tool_id]["function"]["arguments"] = ""
                            tool_calls[tool_id]["function"]["arguments"] += function_data["arguments"]
                            _LOGGER.debug("Appended arguments for tool call %s: %s", tool_id, function_data["arguments"])

                if (
                    tool_id in tool_calls
                    and "name" in tool_calls[tool_id]["function"]
                    and "arguments" in tool_calls[tool_id]["function"]
                    and tool_calls[tool_id]["function"]["arguments"]
                ):
                    try:
                        tool_args = json.loads(tool_calls[tool_id]["function"]["arguments"])
                        _LOGGER.debug("Complete tool call for ID %s: %s", tool_id, tool_calls[tool_id])
                        yield {
                            "tool_calls": [
                                llm.ToolInput(
                                    id=tool_id,
                                    tool_name=tool_calls[tool_id]["function"]["name"],
                                    tool_args=tool_args,
                                )
                            ]
                        }
                        del tool_calls[tool_id]
                        del index_to_id[tool_index]
                    except json.JSONDecodeError as err:
                        _LOGGER.warning("Invalid tool call arguments for ID %s: %s, error: %s", tool_id, tool_calls[tool_id]["function"]["arguments"], err)
                else:
                    _LOGGER.debug("Incomplete tool call for ID %s: %s", tool_id, tool_calls[tool_id])

    if full_response or tool_calls:
        complete_tool_calls = [
            {
                "id": tool_id,
                "function": tool_call["function"],
            }
            for tool_id, tool_call in tool_calls.items()
            if "name" in tool_call["function"] and "arguments" in tool_call["function"]
        ]
        messages.append({
            "role": "assistant",
            "content": full_response,
            "tool_calls": complete_tool_calls,
        })
        _LOGGER.debug("Full LLM response: content=%s, tool_calls=%s", full_response, complete_tool_calls)

class VeniceAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Venice AI conversation entity."""

    _attr_has_entity_name = True
    _attr_name = None
    
    def __init__(
        self,
        entry: ConfigEntry,
        entry_data: AsyncVeniceAIClient,
    ) -> None:
        """Initialize the Venice AI Conversation."""
        _LOGGER.debug("Initializing Venice AI ConversationEntity with entry_id: %s", entry.entry_id)
        super().__init__()
        self.entry = entry
        self.entry_data = entry_data
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="Venice AI Assistant",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        _LOGGER.debug("Adding Venice AI conversation entity to Home Assistant")
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    def _get_exposed_entities(self, chat_log: conversation.ChatLog | None, hass: HomeAssistant) -> list[tuple[str, str | None]]:
        """Retrieve entities and their states exposed to the conversation agent."""
        if not chat_log or not chat_log.llm_api:
            _LOGGER.debug("No LLM API or chat log available; no exposed entities")
            return []
        
        exposed_entities = []
        if hasattr(chat_log.llm_api, "api_prompt") and chat_log.llm_api.api_prompt:
            entity_ids = _parse_api_prompt(chat_log.llm_api.api_prompt)
            for entity_id in entity_ids:
                state_obj = hass.states.get(entity_id)
                state = state_obj.state if state_obj else None
                exposed_entities.append((entity_id, state))
                _LOGGER.debug("Exposed entity: %s, state: %s", entity_id, state)
        
        for tool in chat_log.llm_api.tools:
            if hasattr(tool, "parameters") and isinstance(tool.parameters, dict):
                entity_id = tool.parameters.get("entity_id")
                if entity_id and entity_id not in [e[0] for e in exposed_entities]:
                    state_obj = hass.states.get(entity_id)
                    state = state_obj.state if state_obj else None
                    exposed_entities.append((entity_id, state))
                    _LOGGER.debug("Exposed entity from tool: %s, state: %s", entity_id, state)
        
        _LOGGER.debug("All exposed entities with states: %s", exposed_entities)
        return exposed_entities

    def _get_entity_services(self, chat_log: conversation.ChatLog | None, hass: HomeAssistant) -> dict[str, list[str]]:
        """Retrieve services for exposed entities."""
        if not chat_log or not chat_log.llm_api:
            _LOGGER.debug("No LLM API or chat log available; no entity services")
            return {}
        
        entity_services = {}
        controllable_domains = _get_controllable_domains()
        entities = self._get_exposed_entities(chat_log, hass)
        
        for entity_id, _ in entities:
            domain = entity_id.split(".")[0]
            if domain not in controllable_domains:
                _LOGGER.debug("Entity %s not in controllable domains: %s", entity_id, controllable_domains)
                continue
            if entity_id not in entity_services:
                entity_services[entity_id] = []
            for tool in chat_log.llm_api.tools:
                if tool.name in ["HassTurnOn", "HassTurnOff"]:
                    entity_services[entity_id].append(tool.name)
                elif tool.name == "HassSetPosition" and domain == "cover":
                    entity_services[entity_id].append(tool.name)
                elif tool.name == "HassLightSet" and domain == "light":
                    entity_services[entity_id].append(tool.name)
                elif tool.name == "HassClimateSetTemperature" and domain == "climate":
                    entity_services[entity_id].append(tool.name)
                elif tool.name in ["HassMediaPause", "HassMediaUnpause", "HassMediaNext", "HassMediaPrevious", "HassSetVolume"] and domain == "media_player":
                    entity_services[entity_id].append(tool.name)
                elif tool.name in ["HassVacuumStart", "HassVacuumReturnToBase"] and domain == "vacuum":
                    entity_services[entity_id].append(tool.name)
            _LOGGER.debug("Services for entity %s: %s", entity_id, entity_services[entity_id])
        
        _LOGGER.debug("All entity services: %s", entity_services)
        return entity_services

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the Venice AI API."""
        options = self.entry.options
        client = self.entry_data
        debug_logging = _LOGGER.isEnabledFor(logging.DEBUG)

        _LOGGER.debug(
            "Config entry ID: %s, Options: %s, LLM_HASS_API: %s",
            self.entry.entry_id,
            options,
            options.get(CONF_LLM_HASS_API, "unset")
        )
        _LOGGER.debug(
            "User input: text=%s, conversation_id=%s, agent_id=%s, context_id=%s",
            user_input.text,
            user_input.conversation_id,
            user_input.agent_id,
            user_input.context.id if user_input.context else None
        )

        available_apis = [api.id for api in llm.async_get_apis(self.hass)]
        _LOGGER.debug("Available LLM APIs: %s", available_apis)

        llm_api_id = options.get(CONF_LLM_HASS_API, llm.LLM_API_ASSIST)
        _LOGGER.debug("Attempting to initialize LLM API: %s", llm_api_id)
        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                llm_api_id,
                options.get(CONF_PROMPT, ""),
            )
            if chat_log.llm_api:
                llm_api_attrs = {k: v for k, v in vars(chat_log.llm_api).items() if not k.startswith('_')}
                _LOGGER.debug("LLM API initialized: %s (attributes: %s)", getattr(chat_log.llm_api, 'name', 'unknown'), llm_api_attrs)
            else:
                _LOGGER.debug("LLM API initialized: None")
        except conversation.ConverseError as err:
            _LOGGER.error("Failed to update LLM data for API '%s': %s", llm_api_id, err)
            return err.as_conversation_result()

        if not chat_log.llm_api and llm_api_id == llm.LLM_API_ASSIST and "intent" in available_apis:
            _LOGGER.warning("Assist API failed to initialize; falling back to intent API")
            try:
                await chat_log.async_update_llm_data(
                    DOMAIN,
                    user_input,
                    "intent",
                    options.get(CONF_PROMPT, ""),
                )
                if chat_log.llm_api:
                    llm_api_attrs = {k: v for k, v in vars(chat_log.llm_api).items() if not k.startswith('_')}
                    _LOGGER.debug("Fallback LLM API initialized: %s (attributes: %s)", getattr(chat_log.llm_api, 'name', 'unknown'), llm_api_attrs)
                else:
                    _LOGGER.debug("Fallback LLM API initialized: None")
            except conversation.ConverseError as err:
                _LOGGER.error("Failed to update LLM data for fallback API 'intent': %s", err)
                return err.as_conversation_result()

        if not chat_log.llm_api:
            _LOGGER.warning(
                "No LLM API initialized for '%s'; proceeding without entity control. "
                "Ensure the Assist pipeline is configured with Venice AI (entry_id: %s) as the conversation agent in Settings > Voice Assistants.",
                llm_api_id, self.entry.entry_id
            )

        _LOGGER.debug("ChatLog content: %s", [type(c).__name__ for c in chat_log.content])

        entities = self._get_exposed_entities(chat_log, self.hass)
        services = self._get_entity_services(chat_log, self.hass)
        tools = [tool.name for tool in chat_log.llm_api.tools] if chat_log.llm_api else []
        api_prompt = chat_log.llm_api.api_prompt if chat_log.llm_api and hasattr(chat_log.llm_api, "api_prompt") else ""
        system_content = _format_message_content(
            entities,
            services,
            options.get(CONF_PROMPT, ""),
            api_prompt,
            tools
        )

        if not chat_log.content or not isinstance(chat_log.content[0], conversation.SystemContent):
            async for _ in chat_log.async_add_system_content(system_content):
                pass

        messages = _convert_content(chat_log.content[1:], self.hass)
        messages.insert(0, {"role": "system", "content": system_content})

        if debug_logging:
            _LOGGER.debug("System message sent to Venice AI:\n%s", system_content)
            _LOGGER.debug("Full messages array sent to Venice AI:\n%s", json.dumps(messages, indent=2))

        for iteration in range(MAX_TOOL_ITERATIONS):
            response_generator = client.chat.create(
                model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                messages=messages,
                max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                tools=_get_tool_schema(tools),
                venice_parameters={"include_venice_system_prompt": False},
            )

            try:
                messages.extend(_convert_content([
                    content
                    async for content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id,
                        _transform_stream(response_generator, messages)
                    )
                    if not isinstance(content, conversation.AssistantContent)
                ], self.hass))

                if not chat_log.content:
                    _LOGGER.warning("No content in chat log after processing")
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_speech("No response received from Venice AI")
                    return conversation.ConversationResult(
                        response=intent_response,
                        conversation_id=chat_log.conversation_id,
                        continue_conversation=chat_log.continue_conversation,
                    )

                response_content = chat_log.content[-1]
                if isinstance(response_content, conversation.AssistantContent):
                    response_text = response_content.content or "Action completed"
                    if response_content.tool_calls:
                        _LOGGER.debug("Assistant response with tool calls: %s", response_content.tool_calls)
                    else:
                        parsed_tool_calls = _parse_response_for_tool_call(response_content.content or "", user_input.text, entities)
                        if parsed_tool_calls:
                            _LOGGER.debug("Adding parsed tool calls to response: %s", parsed_tool_calls)
                            # Update messages instead of response_content to avoid FrozenInstanceError
                            messages[-1]["tool_calls"] = parsed_tool_calls
                            # Update response_content by re-adding to chat log
                            async for _ in chat_log.async_add_content(conversation.AssistantContent(
                                content=response_content.content,
                                tool_calls=[
                                    llm.ToolInput(
                                        id=tool_call["id"],
                                        tool_name=tool_call["function"]["name"],
                                        tool_args=json.loads(tool_call["function"]["arguments"]),
                                    )
                                    for tool_call in parsed_tool_calls
                                ]
                            )):
                                pass
                            response_content = chat_log.content[-1]
                elif isinstance(response_content, conversation.ToolResultContent):
                    _LOGGER.debug("Tool result content: %s", response_content)
                    tool_args = json.loads(messages[-1]["tool_calls"][0]["function"]["arguments"])
                    tool_name = messages[-1]["tool_calls"][0]["function"]["name"]
                    entity_name = tool_args.get("name", "Unknown")
                    domain = tool_args.get("domain", "Unknown")
                    action = "turned on" if tool_name == "HassTurnOn" else "turned off"
                    response_text = f"The {entity_name} {domain} has been {action}"
                else:
                    _LOGGER.warning("Last message is not an assistant or tool result message: %s", type(response_content).__name__)
                    intent_response = intent.IntentResponse(language=user_input.language)
                    intent_response.async_set_speech("Failed to process assistant response")
                    return conversation.ConversationResult(
                        response=intent_response,
                        conversation_id=chat_log.conversation_id,
                        continue_conversation=chat_log.continue_conversation,
                    )

            except VeniceAIError as err:
                _LOGGER.error("Error processing with Venice AI: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Error processing with Venice AI: {err}"
                )
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                )

            if not chat_log.unresponded_tool_results:
                break

            if debug_logging:
                _LOGGER.debug("Iteration %d: Unresponded tool results: %s", iteration + 1, chat_log.unresponded_tool_results)

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: Callable[[list[VeniceAIConversationEntity]], None],
) -> None:
    """Set up Venice AI Conversation from a config entry."""
    _LOGGER.debug("Setting up Venice AI conversation entity for entry: %s", entry.entry_id)
    agent = VeniceAIConversationEntity(entry, entry.runtime_data)
    async_add_entities([agent])