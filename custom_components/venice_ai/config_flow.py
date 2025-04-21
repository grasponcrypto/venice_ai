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

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
}

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
            return self.async_create_entry(
                title="Venice AI",
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )

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
        self._config_entry = config_entry

    async def _fetch_models(self) -> list[SelectOptionDict]:
        """Fetch available models from Venice AI."""
        try:
            client = self._config_entry.runtime_data
            models = await client.models.list()
            return [
                SelectOptionDict(
                    value=model["id"],
                    label=model.get("name", model["id"])
                )
                for model in models
            ]
        except Exception as err:
            _LOGGER.error("Failed to fetch models: %s", err)
            return [
                SelectOptionDict(
                    value=RECOMMENDED_CHAT_MODEL,
                    label=f"Default ({RECOMMENDED_CHAT_MODEL})"
                )
            ]

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle options flow."""
        if user_input is not None:
            if user_input.get(CONF_LLM_HASS_API) == "none":
                user_input.pop(CONF_LLM_HASS_API, None)
            if CONF_LLM_HASS_API in user_input and user_input[CONF_LLM_HASS_API] not in [api.id for api in llm.async_get_apis(self.hass)] + ["none"]:
                _LOGGER.warning("Invalid LLM API ID '%s' provided; removing", user_input[CONF_LLM_HASS_API])
                user_input.pop(CONF_LLM_HASS_API, None)
            return self.async_create_entry(title="", data=user_input)

        model_options = await self._fetch_models()
        apis: list[SelectOptionDict] = [
            SelectOptionDict(
                label="No control (disable entity control)",
                value="none",
            )
        ]
        apis.extend(
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(self.hass)
        )

        current_llm_api = self._config_entry.options.get(CONF_LLM_HASS_API, llm.LLM_API_ASSIST)
        if current_llm_api and current_llm_api not in [api["value"] for api in apis]:
            _LOGGER.debug("Current LLM_HASS_API '%s' is invalid; defaulting to '%s'", current_llm_api, llm.LLM_API_ASSIST)
            current_llm_api = llm.LLM_API_ASSIST

        schema = {
            vol.Optional(
                CONF_CHAT_MODEL,
                default=self._config_entry.options.get(
                    CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
                ),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=model_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=self._config_entry.options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=1, max=4096, step=1
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                default=self._config_entry.options.get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=2, step=0.1
                )
            ),
            vol.Optional(
                CONF_TOP_P,
                default=self._config_entry.options.get(
                    CONF_TOP_P, RECOMMENDED_TOP_P
                ),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=1, step=0.1
                )
            ),
            vol.Optional(
                CONF_PROMPT,
                default=self._config_entry.options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                ),
            ): TemplateSelector(),
            vol.Optional(
                "debug_logging",
                default=self._config_entry.options.get("debug_logging", False),
            ): bool,
            vol.Optional(
                CONF_LLM_HASS_API,
                description={
                    "suggested_value": current_llm_api,
                    "description": "Select the LLM API to use for controlling Home Assistant entities."
                },
                default=llm.LLM_API_ASSIST,
            ): SelectSelector(SelectSelectorConfig(options=apis, mode=SelectSelectorMode.DROPDOWN)),
        }

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )