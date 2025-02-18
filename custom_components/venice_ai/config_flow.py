"""Config flow for Venice AI Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

from .client import AsyncVeniceAIClient, VeniceAIError, AuthenticationError
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm, selector
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

class VeniceAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Venice AI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            client = AsyncVeniceAIClient(
                api_key=user_input[CONF_API_KEY],
            )
            await client.models.list()

        except AuthenticationError:
            errors["base"] = "invalid_auth"
        except VeniceAIError:
            errors["base"] = "cannot_connect"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(title="Venice AI", data=user_input)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> VeniceAIOptionsFlow:
        """Get the options flow for this handler."""
        return VeniceAIOptionsFlow(config_entry)


class VeniceAIOptionsFlow(OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        super().__init__()
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors = {}
        models = []

        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        try:
            # Get the client from the config entry's runtime data
            client: AsyncVeniceAIClient = self.config_entry.runtime_data
            models_response = await client.models.list()
            
            # Process models from the API response
            models = [
                {
                    "value": model["id"],
                    "label": f"{model.get('name', model['id'])} ({model['id']})",
                    "disabled": not model.get("model_spec", {}).get("available", True)
                }
                for model in models_response.get("data", [])
            ]
            
            # Sort models by label
            models.sort(key=lambda x: x["label"])

        except Exception as err:
            _LOGGER.error("Error fetching models: %s", err, exc_info=True)
            errors["base"] = "cannot_connect"
            models = [{"value": "default", "label": "Default Model", "disabled": False}]

        # Always include the default model as a fallback
        if not any(model["value"] == "default" for model in models):
            models.insert(0, {"value": "default", "label": "Default Model", "disabled": False})

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(
                    CONF_CHAT_MODEL,
                    default=self.config_entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": model["value"], "label": model["label"]}
                            for model in models
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                )
            }),
            errors=errors,
            description_placeholders={
                "models_info": "\n".join(
                    f"• {model['label']} {'(unavailable)' if model['disabled'] else ''}"
                    for model in models
                )
            }
        )