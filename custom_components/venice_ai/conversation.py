"""Conversation support for Venice AI."""

from collections.abc import AsyncGenerator, Callable
import json
from typing import Any, Literal, cast
import re

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
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# Default system prompt for Venice AI
DEFAULT_SYSTEM_PROMPT = """You are a Home Assistant AI agent powered by Venice AI. Your primary function is to help users control their smart home through natural language commands. You can:

1. Interpret user intents for controlling devices and entities
2. Execute commands through Home Assistant's function calling system
3. Provide status updates about devices and entities
4. Handle multi-turn conversations about home automation

When users make requests:
- Parse their intent carefully
- Use the provided function calls to control entities
- Confirm actions taken
- Be concise but friendly in responses
- If the intent is unclear, do nothing and respond with what is unclear and ask the user to try again
- Use EXACTLY the entity_id as shown in the entity list

When responding with actions:
- Use this exact format for service calls:
service_call
domain.service_name
entity_id

Example response:
The office light will be turned off now.
service_call
light.turn_off
light.office

Keep responses concise but informative.
"""

class VeniceAIConversationEntity(conversation.ConversationEntity):
    """Venice AI conversation entity."""

    _attr_should_expose = True
    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self,
        entry: ConfigEntry,
        entry_data: AsyncVeniceAIClient,
    ) -> None:
        """Initialize the Venice AI Conversation."""
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

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        options = dict(self.entry.options)

        prompt = options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)
        
        # Get only exposed entities through the conversation agent
        exposed_entities = []
        for state in self.hass.states.async_all():
            # Filter entities that are exposed to the conversation agent
            if state.attributes.get("conversation_agent_exposed", False):
                exposed_entities.append(state)
        
        # Format entities text with only exposed entities
        entities_text = "\n".join(
            f"- {state.entity_id}: {state.state}" 
            for state in exposed_entities
        )
        
        # Get available services (synchronous call)
        services = self.hass.services.async_services()
        services_text = []
        for domain, domain_services in services.items():
            for service_name, service in domain_services.items():
                service_desc = f"- {domain}.{service_name}"
                if hasattr(service, "description") and service.description:
                    service_desc += f": {service.description}"
                services_text.append(service_desc)
        
        # Build the system message with HA context
        system_message = (
            f"{prompt}\n\n"
            "Available Home Assistant entities and their states:\n"
            f"{entities_text}\n\n"
            "Available Home Assistant services:\n"
            f"{chr(10).join(services_text)}"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input.text},
        ]

        # Add debug logging
        LOGGER.debug("System message sent to Venice AI:\n%s", system_message)
        LOGGER.debug("User input: %s", user_input.text)

        try:
            response_generator = self.entry_data.chat.create(
                model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                messages=messages,
                max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                venice_parameters={"include_venice_system_prompt": False},
            )
            
            # Collect the full response
            full_response = ""
            async for chunk in response_generator:
                if chunk.choices and chunk.choices[0].get("message", {}).get("content"):
                    full_response += chunk.choices[0]["message"]["content"]
                elif chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
                    full_response += chunk.choices[0]["delta"]["content"]

            # Add response debug
            LOGGER.debug("Full Venice AI response: %s", full_response)

        except VeniceAIError as err:
            raise HomeAssistantError(f"Error processing with Venice AI: {err}") from err

        # Parse and execute service calls
        service_call_match = re.search(
            r"service_call\s+([\w\.]+)\s+([\w\.]+)",
            full_response,
            re.IGNORECASE | re.MULTILINE
        )
        
        if service_call_match:
            try:
                service_full, entity_id = service_call_match.groups()
                domain, service = service_full.split(".", 1)
                
                # Debug logging
                LOGGER.debug("Attempting service call with: domain=%s, service=%s, entity_id=%s", 
                            domain, service, entity_id)
                
                # Verify entity exists
                if not self.hass.states.get(entity_id):
                    LOGGER.error("Entity %s not found. Available entities: %s", 
                                entity_id, 
                                [state.entity_id for state in self.hass.states.async_all() 
                                 if state.entity_id.startswith(domain + ".")])
                    raise ValueError(f"Entity {entity_id} not found")
                
                # Make the service call
                await self.hass.services.async_call(
                    domain,
                    service,
                    {"entity_id": entity_id},
                    blocking=True,
                )
                LOGGER.info("Executed service: %s.%s on %s", domain, service, entity_id)
                
                # Clean the response text
                full_response = re.sub(r"service_call.*?$", "", full_response, flags=re.MULTILINE).strip()
                full_response += f"\n\nSuccessfully executed {service_full} on {entity_id}"
                
            except Exception as err:
                LOGGER.error("Service call failed: %s", err)
                # Add available entities to error message
                available_entities = [
                    state.entity_id 
                    for state in self.hass.states.async_all() 
                    if state.entity_id.startswith(domain + ".")
                ]
                LOGGER.error("Available %s entities: %s", domain, available_entities)
                full_response += f"\n\nError executing action: {str(err)}\nAvailable {domain} entities: {', '.join(available_entities)}"

        # Process the response through the conversation agent
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(full_response)
        
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: Callable[[list[VeniceAIConversationEntity]], None],
) -> None:
    """Set up Venice AI Conversation from a config entry."""
    agent = VeniceAIConversationEntity(entry, entry.runtime_data)
    async_add_entities([agent])