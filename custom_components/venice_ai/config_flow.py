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
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
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

try:
    from .conversation import DEFAULT_SYSTEM_PROMPT
except ImportError:
    LOGGER.warning("Could not import DEFAULT_SYSTEM_PROMPT from conversation.py, using fallback.")
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


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
        """Handle the initial step (API Key entry)."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                LOGGER.debug("Validating Venice AI API key by fetching models")
                client = AsyncVeniceAIClient(api_key=user_input[CONF_API_KEY])
                models_response = await client.models.list()
                if not models_response or not isinstance(models_response, list):
                    raise VeniceAIError("Invalid models response")
                LOGGER.debug("API key validation successful, found %d models", len(models_response))

            except AuthenticationError:
                errors["base"] = "invalid_auth"
                LOGGER.warning("Venice AI authentication failed")
            except VeniceAIError as err:
                errors["base"] = "cannot_connect"
                LOGGER.error("Cannot connect to Venice AI: %s", err)
            except Exception:
                LOGGER.exception("Unexpected exception during Venice AI setup validation")
                errors["base"] = "unknown"
            else:
                initial_options = {
                    CONF_LLM_HASS_API: [],
                    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                    CONF_PROMPT: DEFAULT_SYSTEM_PROMPT,
                    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
                    CONF_TOP_P: RECOMMENDED_TOP_P,
                    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
                    CONF_STRIP_THINKING_RESPONSE: False,
                    CONF_DISABLE_THINKING: False,
                }
                return self.async_create_entry(
                    title="Venice AI", data=user_input, options=initial_options
                )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> VeniceAIOptionsFlow:
        """Get the options flow for this handler."""
        return VeniceAIOptionsFlow(config_entry)


class VeniceAIOptionsFlow(OptionsFlow):
    """Handle options flow for Venice AI integration."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self._client: AsyncVeniceAIClient | None = config_entry.runtime_data

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options for the Venice AI integration."""

        if user_input is not None:
            allowed_ids = {api.id for api in llm.async_get_apis(self.hass)}
            if CONF_LLM_HASS_API in user_input:
                selected_ids = [
                    api_id for api_id in user_input[CONF_LLM_HASS_API] if api_id in allowed_ids
                ]
            else:
                selected_ids = [
                    api_id
                    for api_id in self.config_entry.options.get(CONF_LLM_HASS_API, [])
                    if api_id in allowed_ids
                ]

            updated_options = {**self.config_entry.options, **user_input}
            updated_options[CONF_LLM_HASS_API] = selected_ids

            return self.async_create_entry(title="", data=updated_options)

        # --- Prepare form schema ---
        errors: dict[str, str] = {}
        models_options: list[SelectOptionDict] = [
            SelectOptionDict(value=RECOMMENDED_CHAT_MODEL, label="Default Model")
        ]
        tts_models_options: list[SelectOptionDict] = [
            SelectOptionDict(value=RECOMMENDED_TTS_MODEL, label="TTS Kokoro Model")
        ]
        stt_models_options: list[SelectOptionDict] = [
            SelectOptionDict(value=RECOMMENDED_STT_MODEL, label="NVIDIA Parakeet TDT 0.6B V3")
        ]
        voices_options = [
            SelectOptionDict(value=voice, label=voice)
            for voice in sorted(VENICE_TTS_VOICES)
        ]

        # Fetch models if client is available (best effort)
        if self._client:
            try:
                LOGGER.debug("Fetching text models for options flow")
                models_response = await self._client.models.list(model_type="text")

                if models_response and isinstance(models_response, list):
                    fetched_models = []
                    for model in models_response:
                        model_id = model.get("id")
                        if not model_id:
                            continue

                        model_spec = model.get("model_spec", {})
                        capabilities = model_spec.get("capabilities", {})
                        supports_function_calling = capabilities.get("supportsFunctionCalling", False)

                        if supports_function_calling:
                            model_name = model_spec.get("name", model_id)
                            fetched_models.append(SelectOptionDict(
                                value=model_id,
                                label=f"{model_name} ({model_id})"
                            ))

                    fetched_models.sort(key=lambda x: x["label"])
                    if fetched_models:
                        models_options = fetched_models
                        LOGGER.debug("Found %d text models with function calling support", len(fetched_models))
                    else:
                        LOGGER.warning("No text models with function calling support found")
                else:
                    LOGGER.error("Invalid text models response structure: expected list, got %s", type(models_response))

                # Fetch audio models for TTS and STT
                LOGGER.debug("Fetching audio models for TTS/STT options flow")
                audio_models_response = await self._client.models.list(model_type="audio")

                if audio_models_response and isinstance(audio_models_response, list):
                    fetched_tts_models = []
                    fetched_stt_models = []
                    for model in audio_models_response:
                        model_id = model.get("id")
                        if not model_id:
                            continue

                        model_spec = model.get("model_spec", {})
                        model_name = model_spec.get("name", model_id)
                        capabilities = model_spec.get("capabilities", {})

                        supports_tts = capabilities.get("supportsTTS", False) or "tts" in model_id.lower()
                        supports_stt = capabilities.get("supportsTranscription", False) or "parakeet" in model_id.lower() or "whisper" in model_id.lower()

                        label = f"{model_name} ({model_id})"

                        if supports_tts:
                            fetched_tts_models.append(SelectOptionDict(value=model_id, label=label))
                        if supports_stt:
                            fetched_stt_models.append(SelectOptionDict(value=model_id, label=label))

                    if not fetched_tts_models and not fetched_stt_models:
                        for model in audio_models_response:
                            model_id = model.get("id")
                            if not model_id:
                                continue
                            model_spec = model.get("model_spec", {})
                            model_name = model_spec.get("name", model_id)
                            label = f"{model_name} ({model_id})"

                            if "tts" in model_id.lower() or "kokoro" in model_id.lower():
                                fetched_tts_models.append(SelectOptionDict(value=model_id, label=label))
                            elif "parakeet" in model_id.lower() or "whisper" in model_id.lower() or "stt" in model_id.lower():
                                fetched_stt_models.append(SelectOptionDict(value=model_id, label=label))
                            else:
                                fetched_tts_models.append(SelectOptionDict(value=model_id, label=label))
                                fetched_stt_models.append(SelectOptionDict(value=model_id, label=label))

                    if fetched_tts_models:
                        fetched_tts_models.sort(key=lambda x: x["label"])
                        tts_models_options = fetched_tts_models
                        LOGGER.debug("Found %d TTS models", len(fetched_tts_models))

                    if fetched_stt_models:
                        fetched_stt_models.sort(key=lambda x: x["label"])
                        stt_models_options = fetched_stt_models
                        LOGGER.debug("Found %d STT models", len(fetched_stt_models))

                else:
                    LOGGER.debug("No audio models returned or invalid response structure")

            except AuthenticationError:
                LOGGER.error("Authentication error fetching models for options flow")
                errors["base"] = "invalid_auth"
            except VeniceAIError as err:
                LOGGER.error("Connection error fetching models for options flow: %s", err)
                errors["base"] = "cannot_connect"
            except Exception:
                LOGGER.exception("Unexpected error fetching models for options flow")
                errors["base"] = "unknown"
        else:
            LOGGER.warning("Venice AI client not available in options flow for entry %s. Using default model only.", self.config_entry.entry_id)

        hass_apis = [
            SelectOptionDict(value=api.id, label=api.name)
            for api in llm.async_get_apis(self.hass)
        ]
        llm_api_allowed_ids = {opt["value"] for opt in hass_apis}
        stored_llm_api_ids = self.config_entry.options.get(CONF_LLM_HASS_API, []) or []
        if not isinstance(stored_llm_api_ids, list):
            stored_llm_api_ids = []
        filtered_llm_api_suggested = [
            api_id for api_id in stored_llm_api_ids if api_id in llm_api_allowed_ids
        ]

        options_schema_dict = {
            vol.Optional(
                CONF_CHAT_MODEL,
                default=self.config_entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=models_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_PROMPT,
                default=self.config_entry.options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)
            ): TemplateSelector(),
            vol.Optional(
                CONF_LLM_HASS_API,
                description={"suggested_value": filtered_llm_api_suggested},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=hass_apis,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                default=self.config_entry.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
            ): NumberSelector(
                NumberSelectorConfig(min=0, max=2, step=0.05, mode="slider")
            ),
            vol.Optional(
                CONF_TOP_P,
                default=self.config_entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P)
            ): NumberSelector(
                NumberSelectorConfig(min=0, max=1, step=0.05, mode="slider")
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=self.config_entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
            ): NumberSelector(
                NumberSelectorConfig(min=1, step=1, mode="box")
            ),
            vol.Optional(
                CONF_STRIP_THINKING_RESPONSE,
                default=self.config_entry.options.get(CONF_STRIP_THINKING_RESPONSE, False)
            ): BooleanSelector(),
            vol.Optional(
                CONF_DISABLE_THINKING,
                default=self.config_entry.options.get(CONF_DISABLE_THINKING, False)
            ): BooleanSelector(),
            vol.Optional(
                CONF_TTS_MODEL,
                default=self.config_entry.options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=tts_models_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_VOICE,
                default=self.config_entry.options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=voices_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_RESPONSE_FORMAT,
                default=self.config_entry.options.get(CONF_TTS_RESPONSE_FORMAT, RECOMMENDED_TTS_RESPONSE_FORMAT)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value="mp3", label="MP3"),
                        SelectOptionDict(value="wav", label="WAV"),
                        SelectOptionDict(value="ogg", label="OGG"),
                        SelectOptionDict(value="flac", label="FLAC"),
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_TTS_SPEED,
                default=self.config_entry.options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED)
            ): NumberSelector(
                NumberSelectorConfig(min=0.1, max=3.0, step=0.1, mode="slider")
            ),
            vol.Optional(
                CONF_STT_MODEL,
                default=self.config_entry.options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=stt_models_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_STT_RESPONSE_FORMAT,
                default=self.config_entry.options.get(CONF_STT_RESPONSE_FORMAT, RECOMMENDED_STT_RESPONSE_FORMAT)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value="json", label="JSON"),
                        SelectOptionDict(value="text", label="Text"),
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_STT_TIMESTAMPS,
                default=self.config_entry.options.get(CONF_STT_TIMESTAMPS, RECOMMENDED_STT_TIMESTAMPS)
            ): BooleanSelector(),
        }

        options_schema = vol.Schema(options_schema_dict)

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
        )