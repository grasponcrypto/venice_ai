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
    SelectSelectorMode, # Keep the import for SelectSelectorMode if used elsewhere
    TemplateSelector,
)
# Import client exceptions and client itself
from .client import AsyncVeniceAIClient, AuthenticationError, VeniceAIError
# Import constants for default values and keys
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    # CONF_REASONING_EFFORT, # Not used by Venice
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_STRIP_THINKING_RESPONSE,
    CONF_DISABLE_THINKING,
    CONF_TTS_MODEL,
    CONF_TTS_VOICE,
    CONF_TTS_RESPONSE_FORMAT,
    CONF_TTS_SPEED,
    DOMAIN,
    LOGGER,  # Use existing logger
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    # RECOMMENDED_REASONING_EFFORT, # Not used
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_VOICE,
    RECOMMENDED_TTS_RESPONSE_FORMAT,
    RECOMMENDED_TTS_SPEED,
)
# Import the default prompt from the updated conversation module
try:
    # Ensure this matches the actual location and name
    from .conversation import DEFAULT_SYSTEM_PROMPT
except ImportError:
    # Fallback if conversation.py is not available during import time
    LOGGER.warning("Could not import DEFAULT_SYSTEM_PROMPT from conversation.py, using fallback.")
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


# Schema for the initial setup step (API Key)
STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

# --- Config Flow Handler ---
class VeniceAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Venice AI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step (API Key entry)."""
        errors: dict[str, str] = {}
        if user_input is not None:
            # Validate the API key
            try:
                # Use a temporary client instance for validation
                LOGGER.debug("Validating Venice AI API key by fetching models")
                client = AsyncVeniceAIClient(api_key=user_input[CONF_API_KEY])
                models_response = await client.models.list()
                # Verify we got a valid response - it should be a list of models
                if not models_response or not isinstance(models_response, list):
                    raise VeniceAIError("Invalid models response")
                LOGGER.debug("API key validation successful, found %d models", len(models_response))

            except AuthenticationError:
                errors["base"] = "invalid_auth"
                LOGGER.warning("Venice AI authentication failed")
            except VeniceAIError as err:
                errors["base"] = "cannot_connect"
                LOGGER.error("Cannot connect to Venice AI: %s", err)
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Unexpected exception during Venice AI setup validation")
                errors["base"] = "unknown"
            else:
                # API key is valid, create the entry.
                initial_options = {
                    # Allow selecting multiple LLM APIs (tool providers)
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

        # Show the form again if user_input was None or errors occurred
        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> VeniceAIOptionsFlow:
        """Get the options flow for this handler."""
        return VeniceAIOptionsFlow(config_entry)


# --- Options Flow Handler ---
class VeniceAIOptionsFlow(OptionsFlow):
    """Handle options flow for Venice AI integration."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        # Client instance should be fetched from runtime_data if possible
        self._client: AsyncVeniceAIClient | None = config_entry.runtime_data


    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options for the Venice AI integration."""

        errors: dict[str, str] = {}

        # Handle form submission
        if user_input is not None:
            # If no errors, proceed with saving
            if not errors:
                # Normalize/sanitize selected tool API IDs to avoid stale values
                allowed_ids = {api.id for api in llm.async_get_apis(self.hass)}
                if CONF_LLM_HASS_API in user_input:
                    selected_ids = [
                        api_id for api_id in user_input[CONF_LLM_HASS_API] if api_id in allowed_ids
                    ]
                else:
                    # If not provided by UI, sanitize existing stored selection
                    selected_ids = [
                        api_id
                        for api_id in self.config_entry.options.get(CONF_LLM_HASS_API, [])
                        if api_id in allowed_ids
                    ]

                # Merge new user_input with existing options, prioritizing new input
                updated_options = {**self.config_entry.options, **user_input}
                # Always persist sanitized tool IDs to drop stale ones
                updated_options[CONF_LLM_HASS_API] = selected_ids

                # Create/update the entry with the new options
                return self.async_create_entry(title="", data=updated_options)


        # --- Prepare form schema ---
        models_options: list[SelectOptionDict] = [
              SelectOptionDict(value=RECOMMENDED_CHAT_MODEL, label="Default Model")
        ]
        # Hardcoded list of available voices
        hardcoded_voices = [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
            "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
            "if_sara", "im_nicola",
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
            "pf_dora", "pm_alex", "pm_santa",
            "ef_dora", "em_alex", "em_santa"
        ]
        voices_options = [
            SelectOptionDict(value=voice, label=voice) for voice in sorted(hardcoded_voices)
        ]

        # Fetch models if client is available (best effort)
        if self._client:
            try:
                LOGGER.debug("Fetching models for options flow")
                models_response = await self._client.models.list()

                # Venice AI returns a direct list of models, not {"data": [...]}
                if models_response and isinstance(models_response, list):
                    fetched_models = []
                    for model in models_response:
                        model_id = model.get("id")
                        if not model_id:
                            continue

                        # Check if model supports function calling
                        model_spec = model.get("model_spec", {})
                        capabilities = model_spec.get("capabilities", {})
                        supports_function_calling = capabilities.get("supportsFunctionCalling", False)

                        if supports_function_calling:
                            model_name = model_spec.get("name", model_id)
                            fetched_models.append(SelectOptionDict(
                                value=model_id,
                                label=f"{model_name} ({model_id})"
                            ))

                    # Sort models alphabetically by label
                    fetched_models.sort(key=lambda x: x["label"])
                    # Replace the default model option if we have real models
                    if fetched_models:
                        models_options = fetched_models
                        LOGGER.debug("Found %d models with function calling support", len(fetched_models))
                    else:
                        LOGGER.warning("No models with function calling support found")
                else:
                    LOGGER.error("Invalid models response structure: expected list, got %s", type(models_response))

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
              # Can still show form with default model

        # Get available Home Assistant LLM APIs
        hass_apis = [
            SelectOptionDict(value=api.id, label=api.name)
            for api in llm.async_get_apis(self.hass)
        ]
        # Only suggest values that are still valid options
        llm_api_allowed_ids = {opt["value"] for opt in hass_apis}
        stored_llm_api_ids = self.config_entry.options.get(CONF_LLM_HASS_API, []) or []
        if not isinstance(stored_llm_api_ids, list):
            stored_llm_api_ids = []
        filtered_llm_api_suggested = [
            api_id for api_id in stored_llm_api_ids if api_id in llm_api_allowed_ids
        ]


        # Define the schema for the options form
        # Use current option value as default for the form field
        options_schema_dict = {
            # --- Model Selection ---
            vol.Optional(
                CONF_CHAT_MODEL,
                default=self.config_entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=models_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    # translation_key can be added for frontend i18n
                )
            ),
            # --- Prompt ---
            vol.Optional(
                CONF_PROMPT,
                # Use description for suggested_value if needed, or just default
                default=self.config_entry.options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)
            ): TemplateSelector(),
            # --- LLM HASS API Selection (tools) ---
            vol.Optional(
                CONF_LLM_HASS_API,
                description={"suggested_value": filtered_llm_api_suggested},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=hass_apis,
                    multiple=True,
                )
            ),
            # --- Temperature ---
            vol.Optional(
                CONF_TEMPERATURE,
                default=self.config_entry.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
            ): NumberSelector(
                # Use string literal "slider" for mode
                NumberSelectorConfig(min=0, max=2, step=0.05, mode="slider")
            ),
            # --- Top P ---
            vol.Optional(
                CONF_TOP_P,
                default=self.config_entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P)
            ): NumberSelector(
                # Use string literal "slider" for mode
                NumberSelectorConfig(min=0, max=1, step=0.05, mode="slider")
            ),
            # --- Max Tokens ---
            vol.Optional(
                CONF_MAX_TOKENS,
                default=self.config_entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
            ): NumberSelector(
                # Use string literal "box" for mode
                NumberSelectorConfig(min=1, step=1, mode="box") # Allow integer input
            ),
            # --- Reasoning Model Options ---
            vol.Optional(
                CONF_STRIP_THINKING_RESPONSE,
                default=self.config_entry.options.get(CONF_STRIP_THINKING_RESPONSE, False)
            ): BooleanSelector(),
            vol.Optional(
                CONF_DISABLE_THINKING,
                default=self.config_entry.options.get(CONF_DISABLE_THINKING, False)
            ): BooleanSelector(),
            # --- TTS Model Selection ---
            vol.Optional(
                CONF_TTS_MODEL,
                default=self.config_entry.options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[SelectOptionDict(value=RECOMMENDED_TTS_MODEL, label="TTS Kokoro Model")],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            # --- TTS Voice Selection ---
            vol.Optional(
                CONF_TTS_VOICE,
                default=self.config_entry.options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=voices_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            # --- TTS Response Format ---
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
            # --- TTS Speed ---
            vol.Optional(
                CONF_TTS_SPEED,
                default=self.config_entry.options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED)
            ): NumberSelector(
                NumberSelectorConfig(min=0.1, max=3.0, step=0.1, mode="slider")
            ),
        }

        options_schema = vol.Schema(options_schema_dict)

        # Show the form with the defined schema and potential errors
        # Pass models_info to description_placeholders if needed in frontend strings.json
        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
            # description_placeholders={"models_info": "List models here if needed"}
        )
