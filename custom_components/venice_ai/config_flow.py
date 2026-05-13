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
from homeassistant.helpers.httpx_client import get_async_client
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
    RECOMMENDED_DISABLE_THINKING,
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

    @staticmethod
    async def async_migrate_entry(
        hass: HomeAssistant, entry: ConfigEntry
    ) -> bool:
        """Migrate an old config entry to the current version.

        This is a ``@staticmethod`` matching Home Assistant's core signature.
        Receiving ``self`` is incorrect — when HA calls migration it passes
        only ``hass`` and ``config_entry``.

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

    async def _fetch_model_options(
        self,
    ) -> tuple[list[SelectOptionDict], list[SelectOptionDict], list[SelectOptionDict], dict[str, str]]:
        """Fetch available models from Venice AI and return options + errors.

        The client is scoped entirely within this method via ``async with`` so
        that the underlying aiohttp session is guaranteed to be closed on both
        the happy path and any exception or flow-cancellation path.  Storing
        the client on ``self`` is intentionally avoided: config-flow objects can
        be abandoned between steps, and a ``self._client`` reference that
        survives past the ``finally`` block would leak the session if the flow
        is garbage-collected before cleanup runs.

        Returns:
            (chat_models, tts_models, stt_models, errors)
        """
        chat_options: list[SelectOptionDict] = []
        tts_options: list[SelectOptionDict] = []
        stt_options: list[SelectOptionDict] = []
        errors: dict[str, str] = {}

        api_key = self.config_entry.data.get(CONF_API_KEY)
        if not api_key:
            _LOGGER.warning("No API key found in config entry for options flow")
            errors["base"] = "missing_api_key"
            return chat_options, tts_options, stt_options, errors

        try:
            async with AsyncVeniceAIClient(
                api_key=api_key,
                http_client=get_async_client(self.hass),
            ) as client:
                _LOGGER.debug("Fetching text models for options flow")
                text_resp = await client.models.list(model_type="text")
                if isinstance(text_resp, list):
                    fetched = [
                        SelectOptionDict(label=m.get("id", "Unknown"), value=m.get("id", ""))
                        for m in text_resp
                        if m.get("id")
                    ]
                    if fetched:
                        chat_options = fetched
                        _LOGGER.debug("Found %d text models", len(fetched))
                    else:
                        _LOGGER.warning("No text models found")
                else:
                    _LOGGER.error(
                        "Invalid text models response: expected list, got %s",
                        type(text_resp).__name__,
                    )

                _LOGGER.debug("Fetching TTS models for options flow")
                tts_resp = await client.models.list(model_type="tts")
                if isinstance(tts_resp, list):
                    tts_options = [
                        SelectOptionDict(label=m.get("id", "Unknown"), value=m.get("id", ""))
                        for m in tts_resp
                        if m.get("id")
                    ]
                    _LOGGER.debug("Found %d TTS models", len(tts_options))
                else:
                    _LOGGER.debug("No TTS models returned or invalid response")

                _LOGGER.debug("Fetching ASR models for options flow")
                asr_resp = await client.models.list(model_type="asr")
                if isinstance(asr_resp, list):
                    stt_options = [
                        SelectOptionDict(label=m.get("id", "Unknown"), value=m.get("id", ""))
                        for m in asr_resp
                        if m.get("id")
                    ]
                    _LOGGER.debug("Found %d STT models", len(stt_options))
                else:
                    _LOGGER.debug("No ASR models returned or invalid response")

        except AuthenticationError:
            _LOGGER.error("Authentication error fetching models for options flow")
            errors["base"] = "invalid_auth"
        except VeniceAIError as err:
            _LOGGER.error("Connection error fetching models for options flow: %s", err)
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error fetching models for options flow")
            errors["base"] = "unknown"

        # Fallback to defaults when nothing was fetched
        if not chat_options:
            chat_options = [
                SelectOptionDict(label=RECOMMENDED_CHAT_MODEL, value=RECOMMENDED_CHAT_MODEL)
            ]
        if not tts_options:
            tts_options = [
                SelectOptionDict(label=RECOMMENDED_TTS_MODEL, value=RECOMMENDED_TTS_MODEL)
            ]
        if not stt_options:
            stt_options = [
                SelectOptionDict(label=RECOMMENDED_STT_MODEL, value=RECOMMENDED_STT_MODEL)
            ]

        return chat_options, tts_options, stt_options, errors

    def _build_options_schema(
        self,
        models_options: list[SelectOptionDict],
        tts_models_options: list[SelectOptionDict],
        stt_models_options: list[SelectOptionDict],
        llm_api_options: list[SelectOptionDict] | None = None,
    ) -> vol.Schema:
        """Build the voluptuous options schema from fetched model lists."""
        options = self.config_entry.options
        if llm_api_options is None:
            llm_api_options = [SelectOptionDict(label="None (disabled)", value="")]
        return vol.Schema(
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
                    NumberSelectorConfig(min=1, max=32768, step=1, mode="slider")
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
                    description={"suggested_value": options.get(CONF_LLM_HASS_API, "")},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=llm_api_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=True,
                    )
                ),
                vol.Optional(
                    CONF_STRIP_THINKING_RESPONSE,
                    description={"suggested_value": options.get(CONF_STRIP_THINKING_RESPONSE, False)},
                ): BooleanSelector(),
                vol.Optional(
                    CONF_DISABLE_THINKING,
                    description={"suggested_value": options.get(CONF_DISABLE_THINKING, RECOMMENDED_DISABLE_THINKING)},
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

    async def _fetch_llm_api_options(self) -> list[SelectOptionDict]:
        """Return a list of available HA LLM API IDs as SelectOptionDicts.

        Always includes a leading "None (disabled)" blank entry.  Dynamically
        queries the llm helper if ``async_get_api_list`` is available; falls
        back to the well-known ``"assist"`` API otherwise.
        """
        none_option = SelectOptionDict(label="None (disabled)", value="")
        api_ids: list[str] = []
        try:
            if hasattr(llm, "async_get_api_list"):
                api_ids = await llm.async_get_api_list(self.hass)
            else:
                # Probe the known "assist" API as a safe fallback
                try:
                    await llm.async_get_api(self.hass, "assist")
                    api_ids = ["assist"]
                except Exception:
                    pass
        except Exception:
            _LOGGER.debug("Could not fetch LLM API list; using fallback")
            api_ids = ["assist"]

        return [none_option] + [
            SelectOptionDict(label=api_id, value=api_id) for api_id in api_ids
        ]

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}
        if user_input is not None:
            # Normalise the LLM API field: treat blank string as absent
            if CONF_LLM_HASS_API in user_input and not user_input[CONF_LLM_HASS_API]:
                user_input = {k: v for k, v in user_input.items() if k != CONF_LLM_HASS_API}
            else:
                # SEC-4 fix: validate custom LLM API ID before accepting
                try:
                    await llm.async_get_api(
                        self.hass, user_input[CONF_LLM_HASS_API]
                    )
                except Exception as err:
                    _LOGGER.warning(
                        "Invalid LLM API ID '%s' entered in options flow: %s",
                        user_input[CONF_LLM_HASS_API],
                        err,
                    )
                    errors[CONF_LLM_HASS_API] = "invalid_llm_api"

            if not errors:
                return self.async_create_entry(title="", data=user_input)

        models, tts_models, stt_models, fetch_errors = await self._fetch_model_options()
        llm_api_options = await self._fetch_llm_api_options()
        options_schema = self._build_options_schema(models, tts_models, stt_models, llm_api_options)

        # Merge fetch errors with validation errors
        if fetch_errors:
            errors.update(fetch_errors)

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                options_schema, self.config_entry.options
            ),
            errors=errors,
        )
