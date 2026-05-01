"""Config flow for Venice AI Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm, selector
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from .client import AsyncVeniceAIClient, AuthenticationError, VeniceAIError
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_STRIP_THINKING_RESPONSE,
    CONF_DISABLE_THINKING,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    CONF_TTS_RESPONSE_FORMAT,
    CONF_TTS_SPEED,
    CONF_STT_MODEL,
    CONF_STT_RESPONSE_FORMAT,
    CONF_STT_TIMESTAMPS,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_VOICE,
    RECOMMENDED_TTS_RESPONSE_FORMAT,
    RECOMMENDED_TTS_SPEED,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_STT_RESPONSE_FORMAT,
    RECOMMENDED_STT_TIMESTAMPS,
    VENICE_TTS_VOICES,
)

_LOGGER = logging.getLogger(__name__)

# Try to import DEFAULT_SYSTEM_PROMPT; fallback if not available
try:
    from .conversation import DEFAULT_SYSTEM_PROMPT
except ImportError:
    _LOGGER.warning("Could not import DEFAULT_SYSTEM_PROMPT from conversation.py, using fallback.")
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): cv.string,
    }
)


class VeniceAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Venice AI Conversation."""

    VERSION = 1

    async def async_migrate_entry(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> bool:
        """Migrate an old config entry to the current version.

        Currently there is only version 1, so no migration is needed.
        Future versions should handle data and options migration here.
        """
        if entry.version == 1:
            # Current version — nothing to migrate
            return True
        _LOGGER.error(
            "Unable to migrate config entry from version %s. Please recreate the integration.",
            entry.version,
        )
        return False

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                _LOGGER.debug("Validating Venice AI API key by fetching models")
                async with AsyncVeniceAIClient(api_key=user_input[CONF_API_KEY]) as client:
                    models_response = await client.models.list()
                    if not isinstance(models_response, list):
                        raise VeniceAIError("Invalid models response")

                _LOGGER.debug("API key validation successful, found %d models", len(models_response))

            except AuthenticationError:
                errors["base"] = "invalid_auth"
                _LOGGER.warning("Venice AI authentication failed")
            except VeniceAIError as err:
                errors["base"] = "cannot_connect"
                _LOGGER.error("Cannot connect to Venice AI: %s", err)
            except Exception:
                _LOGGER.exception("Unexpected exception during Venice AI setup validation")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title="Venice AI",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_reauth(
        self, entry_data: MappingProxyType[str, Any]
    ) -> ConfigFlowResult:
        """Handle re-authentication when API key becomes invalid."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm re-authentication with new API key."""
        errors: dict[str, str] = {}
        reauth_entry = self._get_reauth_entry()

        if user_input is not None:
            try:
                _LOGGER.debug("Validating new Venice AI API key for re-auth")
                async with AsyncVeniceAIClient(api_key=user_input[CONF_API_KEY]) as client:
                    models_response = await client.models.list()
                    if not isinstance(models_response, list):
                        raise VeniceAIError("Invalid models response")

                _LOGGER.debug("Re-auth API key validation successful")
            except AuthenticationError:
                errors["base"] = "invalid_auth"
                _LOGGER.warning("Venice AI re-authentication failed: invalid API key")
            except VeniceAIError as err:
                errors["base"] = "cannot_connect"
                _LOGGER.error("Cannot connect to Venice AI during re-auth: %s", err)
            except Exception:
                _LOGGER.exception("Unexpected exception during Venice AI re-auth validation")
                errors["base"] = "unknown"
            else:
                return self.async_update_reload_and_abort(
                    reauth_entry,
                    data={**reauth_entry.data, CONF_API_KEY: user_input[CONF_API_KEY]},
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): cv.string,
                }
            ),
            errors=errors,
            description_placeholders={"name": reauth_entry.title},
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> VeniceAIOptionsFlow:
        """Get the options flow for this handler."""
        return VeniceAIOptionsFlow(config_entry)


class VeniceAIOptionsFlow(OptionsFlow):
    """Options flow for Venice AI."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self._client: AsyncVeniceAIClient | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Build schema with current values as defaults
        options = self.config_entry.options

        # Fetch available models for the selector
        models_options: list[SelectOptionDict] = []
        tts_models_options: list[SelectOptionDict] = []
        stt_models_options: list[SelectOptionDict] = []

        try:
            api_key = self.config_entry.data.get(CONF_API_KEY)
            if api_key:
                self._client = AsyncVeniceAIClient(
                    api_key=api_key,
                )
            else:
                _LOGGER.warning("No API key found in config entry for options flow")
                errors["base"] = "missing_api_key"

            if self._client is not None:
                _LOGGER.debug("Fetching text models for options flow")
                models_response = await self._client.models.list(model_type="text")
                if isinstance(models_response, list):
                    fetched_models = [
                        SelectOptionDict(label=model.get("id", "Unknown"), value=model.get("id", ""))
                        for model in models_response
                        if model.get("id")
                    ]
                    if fetched_models:
                        models_options = fetched_models
                        _LOGGER.debug("Found %d text models with function calling support", len(fetched_models))
                    else:
                        _LOGGER.warning("No text models with function calling support found")
                else:
                    _LOGGER.error("Invalid text models response structure: expected list, got %s", type(models_response))

                # Fetch audio models for TTS and STT
                _LOGGER.debug("Fetching audio models for TTS/STT options flow")
                audio_models_response = await self._client.models.list(model_type="audio")
                if isinstance(audio_models_response, list):
                    tts_models = [m for m in audio_models_response if m.get("id", "").startswith("tts-")]
                    stt_models = [m for m in audio_models_response if m.get("id", "").startswith("stt-") or "parakeet" in m.get("id", "")]

                    fetched_tts_models = [
                        SelectOptionDict(label=model.get("id", "Unknown"), value=model.get("id", ""))
                        for model in tts_models
                        if model.get("id")
                    ]
                    fetched_stt_models = [
                        SelectOptionDict(label=model.get("id", "Unknown"), value=model.get("id", ""))
                        for model in stt_models
                        if model.get("id")
                    ]

                    if fetched_tts_models:
                        tts_models_options = fetched_tts_models
                        _LOGGER.debug("Found %d TTS models", len(fetched_tts_models))

                    if fetched_stt_models:
                        stt_models_options = fetched_stt_models
                        _LOGGER.debug("Found %d STT models", len(fetched_stt_models))

                else:
                    _LOGGER.debug("No audio models returned or invalid response structure")

        except AuthenticationError:
            _LOGGER.error("Authentication error fetching models for options flow")
            errors["base"] = "invalid_auth"
        except VeniceAIError as err:
            _LOGGER.error("Connection error fetching models for options flow: %s", err)
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error fetching models for options flow")
            errors["base"] = "unknown"
        else:
            if not models_options:
                _LOGGER.warning("Venice AI client not available in options flow for entry %s. Using default model only.", self.config_entry.entry_id)
                models_options = [SelectOptionDict(label=RECOMMENDED_CHAT_MODEL, value=RECOMMENDED_CHAT_MODEL)]
            if not tts_models_options:
                tts_models_options = [SelectOptionDict(label=RECOMMENDED_TTS_MODEL, value=RECOMMENDED_TTS_MODEL)]
            if not stt_models_options:
                stt_models_options = [SelectOptionDict(label=RECOMMENDED_STT_MODEL, value=RECOMMENDED_STT_MODEL)]
        finally:
            if self._client is not None:
                await self._client.close()
                self._client = None

        options_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_PROMPT,
                    description={"suggested_value": options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)},
                ): TemplateSelector(),
                vol.Optional(
                    CONF_CHAT_MODEL,
                    description={"suggested_value": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=models_options,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    description={"suggested_value": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)},
                ): NumberSelector(
                    NumberSelectorConfig(min=1, max=4096, step=1, mode="slider")
                ),
                vol.Optional(
                    CONF_TOP_P,
                    description={"suggested_value": options.get(CONF_TOP_P, RECOMMENDED_TOP_P)},
                ): NumberSelector(
                    NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode="slider")
                ),
                vol.Optional(
                    CONF_TEMPERATURE,
                    description={"suggested_value": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)},
                ): NumberSelector(
                    NumberSelectorConfig(min=0.0, max=2.0, step=0.05, mode="slider")
                ),
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={"suggested_value": options.get(CONF_LLM_HASS_API)},
                ): cv.string,
                vol.Optional(
                    CONF_STRIP_THINKING_RESPONSE,
                    description={"suggested_value": options.get(CONF_STRIP_THINKING_RESPONSE, False)},
                ): BooleanSelector(),
                vol.Optional(
                    CONF_DISABLE_THINKING,
                    description={"suggested_value": options.get(CONF_DISABLE_THINKING, False)},
                ): BooleanSelector(),
                vol.Optional(
                    CONF_MAX_TOOL_ITERATIONS,
                    description={"suggested_value": options.get(CONF_MAX_TOOL_ITERATIONS, RECOMMENDED_MAX_TOOL_ITERATIONS)},
                ): NumberSelector(
                    NumberSelectorConfig(min=1, max=20, step=1, mode="slider")
                ),
                # TTS options
                vol.Optional(
                    CONF_TTS_MODEL,
                    description={"suggested_value": options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=tts_models_options,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_TTS_VOICE,
                    description={"suggested_value": options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label=voice, value=voice)
                            for voice in VENICE_TTS_VOICES
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_TTS_RESPONSE_FORMAT,
                    description={"suggested_value": options.get(CONF_TTS_RESPONSE_FORMAT, RECOMMENDED_TTS_RESPONSE_FORMAT)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="MP3", value="mp3"),
                            SelectOptionDict(label="WAV", value="wav"),
                            SelectOptionDict(label="OGG", value="ogg"),
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_TTS_SPEED,
                    description={"suggested_value": options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED)},
                ): NumberSelector(
                    NumberSelectorConfig(min=0.25, max=4.0, step=0.25, mode="slider")
                ),
                # STT options
                vol.Optional(
                    CONF_STT_MODEL,
                    description={"suggested_value": options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=stt_models_options,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_STT_RESPONSE_FORMAT,
                    description={"suggested_value": options.get(CONF_STT_RESPONSE_FORMAT, RECOMMENDED_STT_RESPONSE_FORMAT)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="JSON", value="json"),
                            SelectOptionDict(label="Text", value="text"),
                            SelectOptionDict(label="SRT", value="srt"),
                            SelectOptionDict(label="Verbose JSON", value="verbose_json"),
                            SelectOptionDict(label="VTT", value="vtt"),
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_STT_TIMESTAMPS,
                    description={"suggested_value": options.get(CONF_STT_TIMESTAMPS, RECOMMENDED_STT_TIMESTAMPS)},
                ): BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                options_schema, options
            ),
            errors=errors,
        )
