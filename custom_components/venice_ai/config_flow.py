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
    RECOMMENDED_STRIP_THINKING_RESPONSE,
    CONF_DISABLE_THINKING,
    RECOMMENDED_DISABLE_THINKING,
    CONF_STREAM_RESPONSE,
    RECOMMENDED_STREAM_RESPONSE,
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
    CONF_REQUEST_TIMEOUT,
    RECOMMENDED_REQUEST_TIMEOUT,
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


class _TTSModelInfo:
    """Lightweight container for a TTS model and its voices."""

    def __init__(self, model_id: str, voices: list[str], default_voice: str | None) -> None:
        self.model_id = model_id
        self.voices = voices
        self.default_voice = default_voice


def _resolve_tts_model(
    user_input: dict[str, Any] | None,
    saved_options: dict[str, Any],
    tts_info: dict[str, _TTSModelInfo],
) -> str:
    """Return the TTS model that should drive the voice selector.

    Priority:
    1. A model present in the current form submission.
    2. The previously saved option.
    3. The recommended default.
    """
    if user_input is not None:
        submitted = user_input.get(CONF_TTS_MODEL)
        if isinstance(submitted, str) and submitted in tts_info:
            return submitted

    saved = saved_options.get(CONF_TTS_MODEL)
    if isinstance(saved, str) and saved in tts_info:
        return saved

    if RECOMMENDED_TTS_MODEL in tts_info:
        return RECOMMENDED_TTS_MODEL

    first = next(iter(tts_info))
    return first if isinstance(first, str) else RECOMMENDED_TTS_MODEL


def _resolve_tts_voice(
    selected_model: str,
    tts_info: dict[str, _TTSModelInfo],
    user_input: dict[str, Any] | None,
    saved_options: dict[str, Any],
) -> str:
    """Return the voice that should be pre-selected for a given model.

    Priority:
    1. A voice present in the current form submission (keeps user choice
       when only other fields changed).
    2. The previously saved voice if it is valid for the selected model.
    3. The model's advertised default voice.
    """
    info = tts_info.get(selected_model)
    if info is None:
        return RECOMMENDED_TTS_VOICE

    # Keep user's current selection when the form is re-rendered for
    # unrelated changes (e.g. adjusting temperature).
    if user_input is not None:
        submitted_voice = user_input.get(CONF_TTS_VOICE)
        if isinstance(submitted_voice, str) and submitted_voice in info.voices:
            return submitted_voice

    saved_voice = saved_options.get(CONF_TTS_VOICE)
    if isinstance(saved_voice, str) and saved_voice in info.voices:
        return saved_voice

    if isinstance(info.default_voice, str):
        return info.default_voice
    return info.voices[0] if info.voices else RECOMMENDED_TTS_VOICE


def _extract_tts_model_info(models: list[dict[str, Any]]) -> dict[str, _TTSModelInfo]:
    """Build a lookup of model_id -> _TTSModelInfo from API response objects.

    Venice AI returns voices under ``model_spec.voices``.  The legacy
    ``voice_models`` field is also honoured as a fallback.
    """
    info: dict[str, _TTSModelInfo] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue

        voices: list[str] = []

        # Primary source: model_spec.voices (observed live API shape).
        raw_spec = model.get("model_spec")
        model_spec: dict[str, Any] = raw_spec if isinstance(raw_spec, dict) else {}
        spec_voices = model_spec.get("voices")
        if isinstance(spec_voices, list):
            voices = [str(v) for v in spec_voices if isinstance(v, str) and v]

        # Fallback: legacy voice_models field used by earlier implementations.
        if not voices:
            voice_models = model.get("voice_models")
            if isinstance(voice_models, list):
                voices = [str(v) for v in voice_models if isinstance(v, str) and v]

        if not voices:
            continue

        # Prefer an explicit default_voice if provided, otherwise the first voice.
        default_voice = model.get("default_voice")
        if not isinstance(default_voice, str) or default_voice not in voices:
            default_voice = voices[0]

        info[model_id] = _TTSModelInfo(str(model_id), voices, default_voice)

    return info


def _build_voice_options(
    selected_model: str,
    tts_info: dict[str, _TTSModelInfo],
) -> list[SelectOptionDict]:
    """Return SelectOptionDict voices for the selected TTS model."""
    info = tts_info.get(selected_model)
    if info is None or not info.voices:
        return [SelectOptionDict(label=RECOMMENDED_TTS_VOICE, value=RECOMMENDED_TTS_VOICE)]
    return [SelectOptionDict(label=voice, value=voice) for voice in info.voices]


class VeniceAIOptionsFlow(OptionsFlow):
    """Options flow for Venice AI.

    All settings — including TTS model and voice — are shown on a **single
    form**.  All model/voice data is fetched once when the form opens and
    cached for the session.

    When the user changes the TTS model and submits, the flow detects that
    the currently shown voice may not be valid for the new model and
    re-renders the same form with the voice dropdown re-populated with only
    voices compatible with the newly selected model (defaulting to that
    model's default voice).  The user is never presented with an invalid
    pairing and never needs to navigate to a second screen.

    If the TTS model is unchanged the form saves immediately on submit.
    """

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        super().__init__()
        self._config_entry = config_entry
        # TTS metadata fetched on first open and reused on re-renders (no extra API calls).
        self._tts_info: dict[str, _TTSModelInfo] = {}
        # The TTS model that was shown on the last render, used to detect model changes.
        self._shown_tts_model: str | None = None

    @property
    def config_entry(self) -> ConfigEntry:
        """Return the config entry."""
        return self._config_entry

    def _extract_tts_model_info(self, models: list[dict[str, Any]]) -> dict[str, _TTSModelInfo]:
        """Build a lookup of model_id -> _TTSModelInfo from API response objects."""
        return _extract_tts_model_info(models)

    async def _fetch_model_metadata(
        self,
    ) -> tuple[
        list[SelectOptionDict],
        dict[str, _TTSModelInfo],
        list[SelectOptionDict],
        dict[str, str],
    ]:
        """Fetch the latest available models live from Venice AI.

        A fresh API call is always made when the options flow opens so that
        any models Venice.ai has added since the last coordinator refresh are
        immediately visible.  The coordinator's hourly cache is intentionally
        bypassed here — the options flow is only opened on user demand and
        correctness matters more than saving one lightweight API call.

        Returns:
            (chat_model_options, tts_model_info, stt_model_options, errors)
        """
        chat_options: list[SelectOptionDict] = []
        tts_info: dict[str, _TTSModelInfo] = {}
        stt_options: list[SelectOptionDict] = []
        errors: dict[str, str] = {}

        api_key = self.config_entry.data.get(CONF_API_KEY)
        if not api_key:
            _LOGGER.warning("No API key found in config entry for options flow")
            errors["base"] = "missing_api_key"
            return chat_options, tts_info, stt_options, errors

        try:
            async with AsyncVeniceAIClient(
                api_key=api_key,
                http_client=get_async_client(self.hass),
            ) as client:
                _LOGGER.debug("Fetching text models for options flow")
                text_resp = await client.models.list(model_type="text")
                if isinstance(text_resp, list):
                    chat_options = [
                        SelectOptionDict(label=m.get("id", "Unknown"), value=m.get("id", ""))
                        for m in text_resp
                        if isinstance(m, dict) and m.get("id")
                    ]
                    _LOGGER.debug("Found %d text models", len(chat_options))

                _LOGGER.debug("Fetching TTS models for options flow")
                tts_resp = await client.models.list(model_type="tts")
                if isinstance(tts_resp, list):
                    tts_info = self._extract_tts_model_info(tts_resp)
                    _LOGGER.debug("Found %d TTS models with voices", len(tts_info))

                _LOGGER.debug("Fetching ASR models for options flow")
                asr_resp = await client.models.list(model_type="asr")
                if isinstance(asr_resp, list):
                    stt_options = [
                        SelectOptionDict(label=m.get("id", "Unknown"), value=m.get("id", ""))
                        for m in asr_resp
                        if isinstance(m, dict) and m.get("id")
                    ]
                    _LOGGER.debug("Found %d STT models", len(stt_options))

        except AuthenticationError:
            _LOGGER.error("Authentication error fetching models for options flow")
            errors["base"] = "invalid_auth"
        except VeniceAIError as err:
            _LOGGER.error("Connection error fetching models for options flow: %s", err)
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error fetching models for options flow")
            errors["base"] = "unknown"

        # Fallback to defaults when nothing was fetched.
        if not chat_options:
            chat_options = [
                SelectOptionDict(label=RECOMMENDED_CHAT_MODEL, value=RECOMMENDED_CHAT_MODEL)
            ]
        if not tts_info:
            tts_info = {
                RECOMMENDED_TTS_MODEL: _TTSModelInfo(
                    RECOMMENDED_TTS_MODEL,
                    [RECOMMENDED_TTS_VOICE],
                    RECOMMENDED_TTS_VOICE,
                )
            }
        if not stt_options:
            stt_options = [
                SelectOptionDict(label=RECOMMENDED_STT_MODEL, value=RECOMMENDED_STT_MODEL)
            ]

        return chat_options, tts_info, stt_options, errors

    def _resolve_tts_model(
        self,
        user_input: dict[str, Any] | None,
        tts_info: dict[str, _TTSModelInfo],
    ) -> str:
        """Return the TTS model that should drive the voice selector."""
        return _resolve_tts_model(user_input, self.config_entry.options, tts_info)

    def _resolve_tts_voice(
        self,
        selected_model: str,
        tts_info: dict[str, _TTSModelInfo],
        user_input: dict[str, Any] | None,
    ) -> str:
        """Return the voice that should be pre-selected for a given model."""
        return _resolve_tts_voice(
            selected_model, tts_info, user_input, self.config_entry.options
        )

    def _build_voice_options(
        self,
        selected_model: str,
        tts_info: dict[str, _TTSModelInfo],
    ) -> list[SelectOptionDict]:
        """Return SelectOptionDict voices for the selected TTS model."""
        return _build_voice_options(selected_model, tts_info)

    def _build_options_schema(
        self,
        chat_options: list[SelectOptionDict],
        tts_info: dict[str, _TTSModelInfo],
        stt_options: list[SelectOptionDict],
        selected_tts_model: str,
        llm_api_options: list[SelectOptionDict] | None = None,
    ) -> vol.Schema:
        """Build the full options schema including TTS model and filtered voice list.

        The voice dropdown is built from the voices available for
        ``selected_tts_model``, so it always shows only valid choices.
        When the user picks a different model and submits, ``async_step_init``
        re-renders this schema with the new model's voices pre-populated.

        Suggested values are supplied by the caller via
        ``add_suggested_values_to_schema``.
        """
        tts_model_options = [
            SelectOptionDict(label=model_id, value=model_id)
            for model_id in sorted(tts_info)
        ] or [SelectOptionDict(label=RECOMMENDED_TTS_MODEL, value=RECOMMENDED_TTS_MODEL)]

        voice_options = _build_voice_options(selected_tts_model, tts_info)

        if llm_api_options is None:
            llm_api_options = [SelectOptionDict(label="None (disabled)", value="")]

        return vol.Schema(
            {
                vol.Optional(CONF_PROMPT): TemplateSelector(),
                vol.Optional(CONF_CHAT_MODEL): SelectSelector(
                    SelectSelectorConfig(
                        options=chat_options,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_MAX_TOKENS): NumberSelector(
                    NumberSelectorConfig(min=1, max=32768, step=1, mode="slider")
                ),
                vol.Optional(CONF_TOP_P): NumberSelector(
                    NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode="slider")
                ),
                vol.Optional(CONF_TEMPERATURE): NumberSelector(
                    NumberSelectorConfig(min=0.0, max=2.0, step=0.05, mode="slider")
                ),
                vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                    SelectSelectorConfig(
                        options=llm_api_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=True,
                    )
                ),
                vol.Optional(CONF_STRIP_THINKING_RESPONSE): BooleanSelector(),
                vol.Optional(CONF_DISABLE_THINKING): BooleanSelector(),
                vol.Optional(CONF_STREAM_RESPONSE): BooleanSelector(),
                vol.Optional(CONF_MAX_TOOL_ITERATIONS): NumberSelector(
                    NumberSelectorConfig(min=1, max=20, step=1, mode="slider")
                ),
                # TTS — model and voice on the same form.
                # Voice options are filtered to the selected model; changing
                # the model and submitting re-renders the form with updated voices.
                vol.Optional(CONF_TTS_MODEL): SelectSelector(
                    SelectSelectorConfig(
                        options=tts_model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_TTS_VOICE): SelectSelector(
                    SelectSelectorConfig(
                        options=voice_options,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_TTS_RESPONSE_FORMAT): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label="MP3", value="mp3"),
                            SelectOptionDict(label="WAV", value="wav"),
                            SelectOptionDict(label="OGG", value="ogg"),
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_TTS_SPEED): NumberSelector(
                    NumberSelectorConfig(min=0.25, max=4.0, step=0.25, mode="slider")
                ),
                # STT options
                vol.Optional(CONF_STT_MODEL): SelectSelector(
                    SelectSelectorConfig(
                        options=stt_options,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_STT_RESPONSE_FORMAT): SelectSelector(
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
                vol.Optional(CONF_STT_TIMESTAMPS): BooleanSelector(),
                vol.Optional(CONF_REQUEST_TIMEOUT): NumberSelector(
                    NumberSelectorConfig(min=10.0, max=300.0, step=5.0, mode="slider")
                ),
            }
        )

    async def _fetch_llm_api_options(self) -> list[SelectOptionDict]:
        """Return a list of available HA LLM API IDs as SelectOptionDicts."""
        none_option = SelectOptionDict(label="None (disabled)", value="")
        api_ids: list[str] = []
        try:
            if hasattr(llm, "async_get_api_list"):
                api_ids = await llm.async_get_api_list(self.hass)
            else:
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

    def _validate_numeric_options(
        self,
        user_input: dict[str, Any],
    ) -> dict[str, str]:
        """ARCH-4: validate numeric option ranges and return per-field error keys."""
        errors: dict[str, str] = {}
        max_tokens = user_input.get(CONF_MAX_TOKENS)
        if isinstance(max_tokens, (int, float)) and not (1 <= max_tokens <= 32768):
            errors[CONF_MAX_TOKENS] = "max_tokens_out_of_range"
        top_p = user_input.get(CONF_TOP_P)
        if isinstance(top_p, (int, float)) and not (0.0 <= top_p <= 1.0):
            errors[CONF_TOP_P] = "top_p_out_of_range"
        temperature = user_input.get(CONF_TEMPERATURE)
        if isinstance(temperature, (int, float)) and not (0.0 <= temperature <= 2.0):
            errors[CONF_TEMPERATURE] = "temperature_out_of_range"
        max_tool_iter = user_input.get(CONF_MAX_TOOL_ITERATIONS)
        if isinstance(max_tool_iter, (int, float)) and not (1 <= max_tool_iter <= 20):
            errors[CONF_MAX_TOOL_ITERATIONS] = "max_tool_iterations_out_of_range"
        tts_speed = user_input.get(CONF_TTS_SPEED)
        if isinstance(tts_speed, (int, float)) and not (0.25 <= tts_speed <= 4.0):
            errors[CONF_TTS_SPEED] = "tts_speed_out_of_range"
        timeout = user_input.get(CONF_REQUEST_TIMEOUT)
        if isinstance(timeout, (int, float)) and not (10.0 <= timeout <= 300.0):
            errors[CONF_REQUEST_TIMEOUT] = "request_timeout_out_of_range"
        prompt = user_input.get(CONF_PROMPT)
        if prompt is not None and not isinstance(prompt, str):
            errors[CONF_PROMPT] = "prompt_must_be_string"
        return errors

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Single-step options form: all settings including TTS model and voice.

        All model/voice data is fetched once when the form opens (and cached
        for any re-renders so no extra API calls are made).

        On submission the flow checks whether the submitted voice is valid for
        the submitted TTS model:

        * If the **TTS model changed** since the last render and the submitted
          voice is no longer valid, the form re-renders with the voice dropdown
          re-populated with only voices for the new model and the model's
          default voice pre-selected.  The user sees the updated list
          immediately and submits again — no second page required.

        * If the voice is valid for the selected model (model unchanged, or
          user already picked a valid voice for the new model) the options are
          saved immediately.
        """
        errors: dict[str, str] = {}

        # Always fetch live from the Venice AI API so newly added models and
        # voices are immediately visible when the user opens the options form.
        chat_options, tts_info, stt_options, fetch_errors = await self._fetch_model_metadata()
        llm_api_options = await self._fetch_llm_api_options()
        self._tts_info = tts_info

        if fetch_errors:
            errors.update(fetch_errors)

        # The TTS model that drives the voice dropdown for this render.
        selected_tts_model = self._resolve_tts_model(user_input, tts_info)

        if user_input is not None and not errors:
            # --- Normalise LLM API field ---
            if CONF_LLM_HASS_API in user_input and not user_input[CONF_LLM_HASS_API]:
                user_input = {k: v for k, v in user_input.items() if k != CONF_LLM_HASS_API}
            elif CONF_LLM_HASS_API in user_input and user_input[CONF_LLM_HASS_API]:
                try:
                    await llm.async_get_api(self.hass, user_input[CONF_LLM_HASS_API])
                except Exception as err:
                    _LOGGER.warning(
                        "Invalid LLM API ID '%s': %s",
                        user_input[CONF_LLM_HASS_API],
                        err,
                    )
                    errors[CONF_LLM_HASS_API] = "invalid_llm_api"

            # --- Validate numeric ranges ---
            errors.update(self._validate_numeric_options(user_input))

            if not errors:
                submitted_voice = user_input.get(CONF_TTS_VOICE)
                model_info = tts_info.get(selected_tts_model)
                voice_valid = (
                    model_info is not None
                    and isinstance(submitted_voice, str)
                    and submitted_voice in model_info.voices
                )
                model_changed = (
                    self._shown_tts_model is not None
                    and self._shown_tts_model != selected_tts_model
                )

                if not voice_valid and model_changed:
                    # The user switched the TTS model; the voice they had
                    # selected is not valid for the new model.  Re-render
                    # the same form with the new model's voices and default
                    # voice pre-selected.  The user submits once more to save.
                    _LOGGER.debug(
                        "TTS model changed to '%s'; re-rendering with updated voices",
                        selected_tts_model,
                    )
                    # Fall through to re-render below (user_input is retained
                    # for all other fields via suggested_values).
                elif not voice_valid:
                    # Model didn't change but the voice somehow isn't in the
                    # list (e.g. stale value).  Treat as a field error.
                    errors[CONF_TTS_VOICE] = "invalid_tts_voice_for_model"
                else:
                    # Voice is valid → save immediately.
                    final_options = dict(user_input)
                    final_options[CONF_TTS_MODEL] = selected_tts_model
                    final_options[CONF_TTS_VOICE] = submitted_voice
                    _LOGGER.debug(
                        "Options saved: TTS model='%s', voice='%s'",
                        selected_tts_model,
                        submitted_voice,
                    )
                    return self.async_create_entry(title="", data=final_options)

        # --- Build the schema for this render ---
        # Record which model is shown so the next submission can detect changes.
        self._shown_tts_model = selected_tts_model

        options_schema = self._build_options_schema(
            chat_options,
            tts_info,
            stt_options,
            selected_tts_model,
            llm_api_options,
        )

        # Suggested values: defaults → saved options → current submission.
        # The voice is resolved to the best choice for the current model.
        suggested_values: dict[str, Any] = {
            CONF_PROMPT: DEFAULT_SYSTEM_PROMPT,
            CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
            CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
            CONF_TOP_P: RECOMMENDED_TOP_P,
            CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
            CONF_LLM_HASS_API: "",
            CONF_STRIP_THINKING_RESPONSE: RECOMMENDED_STRIP_THINKING_RESPONSE,
            CONF_DISABLE_THINKING: RECOMMENDED_DISABLE_THINKING,
            CONF_STREAM_RESPONSE: RECOMMENDED_STREAM_RESPONSE,
            CONF_MAX_TOOL_ITERATIONS: RECOMMENDED_MAX_TOOL_ITERATIONS,
            CONF_TTS_MODEL: selected_tts_model,
            CONF_TTS_RESPONSE_FORMAT: RECOMMENDED_TTS_RESPONSE_FORMAT,
            CONF_TTS_SPEED: RECOMMENDED_TTS_SPEED,
            CONF_STT_MODEL: RECOMMENDED_STT_MODEL,
            CONF_STT_RESPONSE_FORMAT: RECOMMENDED_STT_RESPONSE_FORMAT,
            CONF_STT_TIMESTAMPS: RECOMMENDED_STT_TIMESTAMPS,
            CONF_REQUEST_TIMEOUT: RECOMMENDED_REQUEST_TIMEOUT,
        }
        suggested_values.update(self.config_entry.options)
        if user_input is not None:
            suggested_values.update(user_input)

        # Always pin the resolved model and a valid voice for it.
        suggested_values[CONF_TTS_MODEL] = selected_tts_model
        suggested_values[CONF_TTS_VOICE] = _resolve_tts_voice(
            selected_tts_model, tts_info, user_input, self.config_entry.options
        )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                options_schema, suggested_values
            ),
            errors=errors,
        )
