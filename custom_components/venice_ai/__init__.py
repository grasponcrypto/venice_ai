   # venice_ai/__init__.py

from __future__ import annotations

import logging
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.components import conversation 
from .const import (
    DOMAIN,
    CONF_API_KEY,
    CONF_NAME,
    CONF_MODEL,
    DEFAULT_MODEL,
    DEFAULT_BASE_URL
)
import aiohttp
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent
import json

_LOGGER = logging.getLogger(__name__)

# Use the correct function to create the config schema
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up Venice AI Conversation."""
    return True

async def async_setup_entry(hass: HomeAssistant, entry: config_entries.ConfigEntry) -> bool:
    """Set up Venice AI Conversation from a config entry."""
    try:
        api_key = entry.data[CONF_API_KEY]
        base_url = DEFAULT_BASE_URL  # Use constant
        model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
        name = entry.data.get(CONF_NAME, "Venice AI")

        # Validate API key
        await validate_authentication(api_key, base_url)

        # Create and register agent
        agent = VeniceAIConversationAgent(hass, entry, name)
        conversation.async_set_agent(hass, entry, agent)

        # Store the entry data in hass.data
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "agent": agent,
            "name": name
        }

    except Exception as e:
        _LOGGER.error("Failed to set up entry: %s", e)
        raise ConfigEntryNotReady from e

    return True

async def async_unload_entry(hass: HomeAssistant, entry: config_entries.ConfigEntry) -> bool:
    """Unload Venice AI Conversation."""
    conversation.async_unset_agent(hass, entry)
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True

async def validate_authentication(api_key: str, base_url: str) -> None:
    """Validate the API key with the Venice AI service."""
    if not api_key:
        raise ValueError("Invalid API Key")

async def fetch_models(api_key: str, base_url: str) -> list:
    """Fetch the list of models from the Venice AI API."""
    url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()  # Return the list of models
            else:
                _LOGGER.error("Failed to fetch models: %s", await response.text())
                return []

class VeniceAIConversationAgent(conversation.AbstractConversationAgent):
    """Venice AI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: config_entries.ConfigEntry, name: str) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self._attr_name = name
        self._attr_unique_id = entry.entry_id
        self._attr_supported_languages = "*"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=name,
            manufacturer="Venice AI"
        )

    @property
    def attribution(self) -> str | None:
        return "Powered by Venice AI"

    @property
    def supported_languages(self) -> list[str] | str:
        """Return list of supported languages."""
        return "*"  # Support all languages

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a conversation query."""
        try:
            api_key = self.entry.data[CONF_API_KEY]
            base_url = DEFAULT_BASE_URL  # Single source
            model = self.entry.options.get(CONF_MODEL, DEFAULT_MODEL)
            
            # Direct endpoint construction
            endpoint = f"{base_url}/chat/completions"
            _LOGGER.debug("Attempting API call to endpoint: %s", endpoint)
            _LOGGER.debug("Using model: %s", model)
            
            async with aiohttp.ClientSession(
                json_serialize=json.dumps,
                headers={"Accept": "application/json"}  # Explicitly request JSON
            ) as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": user_input.text}],
                    "temperature": 0.7,
                    "stream": False
                }
                
                # Debug logging for request details
                _LOGGER.debug("Request headers: %s", headers)
                _LOGGER.debug("Request payload: %s", payload)
                
                # Add debug logging for raw response
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    raw_response = await response.text()
                    _LOGGER.debug("Raw API response: %s", raw_response)
                    
                    try:
                        result = await response.json(content_type=None)  # Allow any content type
                    except json.JSONDecodeError:
                        _LOGGER.error("Failed to parse JSON response: %s", raw_response)
                        raise HomeAssistantError("Invalid JSON response from API")

                    _LOGGER.debug("API Response: %s", result)  # Debug logging

                    # Validate response structure
                    if not isinstance(result.get("choices"), list) or len(result["choices"]) == 0:
                        raise HomeAssistantError("Invalid API response format - missing choices array")

                    choice = result["choices"][0]
                    if "message" not in choice or "content" not in choice["message"]:
                        raise HomeAssistantError("Malformed message in API response")

                    response_text = choice["message"]["content"]

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response_text)
            return conversation.ConversationResult(
                response=intent_response, 
                conversation_id=user_input.conversation_id
            )

        except aiohttp.ClientError as err:
            _LOGGER.error("Connection error: %s", err)
            raise HomeAssistantError(f"Connection error: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Invalid JSON response: %s", err)
            raise HomeAssistantError("Invalid API response format") from err
        except KeyError as err:
            _LOGGER.error("Missing expected field in response: %s", err)
            raise HomeAssistantError("Unexpected API response structure") from err
        except Exception as err:
            _LOGGER.error("Unexpected error: %s", err)
            raise HomeAssistantError(f"Processing error: {err}") from err