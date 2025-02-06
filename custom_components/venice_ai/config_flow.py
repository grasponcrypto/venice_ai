from __future__ import annotations

import logging
import voluptuous as vol
import aiohttp
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from .const import (
    DOMAIN,
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    CONF_MODEL,
    CONF_API_KEY,
    CONF_NAME
)

_LOGGER = logging.getLogger(__name__)

# Define your schema for user input
STEP_USER_DATA_SCHEMA = vol.Schema({
    vol.Required(CONF_API_KEY): str,
    vol.Optional(CONF_NAME, default="Venice AI"): str,
})

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
                models_data = await response.json()
                return [model['id'] for model in models_data['data']]  # Adjust based on actual response structure
            else:
                _LOGGER.error("Failed to fetch models: %s", await response.text())
                return []

async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate user input."""
    api_key = data[CONF_API_KEY]
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{DEFAULT_BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"API Error {response.status}: {error_text}")

class VeniceAIConversationConfigFlow(config_entries.ConfigFlow, domain="venice_ai"):
    """Handle a config flow for Venice AI."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial step."""
        errors = {}
        
        if user_input is not None:
            try:
                await validate_input(self.hass, user_input)
                return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)
            except ValueError as e:
                if "API Error" in str(e):
                    errors["base"] = "invalid_auth"
                else:
                    errors["base"] = "unknown"
                _LOGGER.error("Validation error: %s", e)
        
        return self.async_show_form(
            step_id="user", 
            data_schema=STEP_USER_DATA_SCHEMA, 
            errors=errors
        )

    async def async_step_configure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the options step."""
        errors = {}
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        
        if user_input is not None:
            try:
                # Verify model is valid
                await self._validate_model(
                    entry.data[CONF_API_KEY],
                    entry.data[CONF_BASE_URL],
                    user_input[CONF_MODEL]
                )
                return self.async_create_entry(title="", data=user_input)
            except ValueError as err:
                errors["base"] = "invalid_model"
                _LOGGER.error("Model validation failed: %s", err)

        # Get available models
        try:
            models = await self._fetch_models(
                entry.data[CONF_API_KEY],
                entry.data[CONF_BASE_URL]
            )
        except Exception as err:
            errors["base"] = "model_fetch_failed"
            models = []
            _LOGGER.error("Failed to fetch models: %s", err)

        return self.async_show_form(
            step_id="configure",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_MODEL,
                    default=entry.options.get(CONF_MODEL, DEFAULT_MODEL)
                ): vol.In(models) if models else str
            }),
            description_placeholders={"note": "⚠️ Models fetched from Venice AI API"},
            errors=errors
        )

    async def _fetch_models(self, api_key: str, base_url: str) -> list[str]:
        """Fetch available models from API."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            ) as response:
                if response.status != 200:
                    raise ValueError(f"API Error {response.status}")
                data = await response.json()
                return [model["id"] for model in data.get("data", [])]

    async def _validate_model(self, api_key: str, base_url: str, model: str) -> None:
        """Validate model exists in available models."""
        models = await self._fetch_models(api_key, base_url)
        if model not in models:
            raise ValueError(f"Model {model} not in available models: {models}")

    # Optionally, implement an options flow if needed
