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

# OptionsFlowWithReload was introduced in HA 2024.1 and automatically reloads
# the integration when options are saved, removing the need for a manual
# add_update_listener in __init__.py.  We fall back to plain OptionsFlow so
# the integration still loads on older cores (manifest minimum is 2024.4.0,
# so the try-branch will always win in practice).
try:
    from homeassistant.config_entries import OptionsFlowWithReload as _OptionsFlowBase
except ImportError:  # pragma: no cover – only hit on very old HA cores
    _OptionsFlowBase = OptionsFlow  # type: ignore[assignment, misc]
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

# ---------------------------------------------------------------------------
# Combined TTS model + voice selector
# ---------------------------------------------------------------------------
# Instead of two separate dropdowns (model, then voice) that require a
# re-render handshake to stay in sync, we expose a **single** searchable
# dropdown whose entries look like:
#
#   "kokoro → af_heart"
#   "kokoro → af_sky"
#   "outertts → nova"
#
# The user types a model name to filter, picks an entry, and both model and
# voice are set atomically in one submit.  No second page, no silent
# re-render, no abandonment risk.
#
# On save the combined value is split back into CONF_TTS_MODEL / CONF_TTS_VOICE
# for storage — the rest of the integration is unchanged.
# ---------------------------------------------------------------------------

# Visual separator used inside combined values.  Space-arrow-space is easy to
# read and extremely unlikely to appear inside a Venice AI model or voice ID.
_TTS_MV_SEP = " → "

# Form-only schema key.  This key is **never** written to config entry options;
# it is parsed on submit and stored as CONF_TTS_MODEL + CONF_TTS_VOICE.
_CONF_TTS_MODEL_VOICE = "tts_model_voice"

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


# ---------------------------------------------------------------------------
# TTS model metadata helpers
# ---------------------------------------------------------------------------

class _TTSModelInfo:
    """Lightweight container for a TTS model and its voices."""

    def __init__(self, model_id: str, voices: list[str], default_voice: str | None) -> None:
        self.model_id = model_id
        self.voices = voices
        self.default_voice = default_voice


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


def _build_combined_tts_options(
    tts_info: dict[str, _TTSModelInfo],
) -> list[SelectOptionDict]:
    """Return a flat 'model → voice' SelectOptionDict list covering all models.

    Options are sorted by model name so entries for the same model are grouped
    together in the dropdown.  Because HA renders this in dropdown mode with a
    live search box, the user can type a model name to instantly filter to only
    that model's voices, or type a voice name to find it across all models.
    """
    options: list[SelectOptionDict] = []
    for model_id in sorted(tts_info):
        info = tts_info[model_id]
        for voice in info.voices:
            combined = f"{model_id}{_TTS_MV_SEP}{voice}"
            options.append(SelectOptionDict(label=combined, value=combined))
    if not options:
        fallback = f"{RECOMMENDED_TTS_MODEL}{_TTS_MV_SEP}{RECOMMENDED_TTS_VOICE}"
        options = [SelectOptionDict(label=fallback, value=fallback)]
    return options


def _parse_combined_tts_value(value: str) -> tuple[str, str] | None:
    """Split a 'model → voice' string into (model_id, voice).

    Returns ``None`` if the value cannot be parsed into two non-empty parts.
    """
    if _TTS_MV_SEP in value:
        model_id, _, voice = value.partition(_TTS_MV_SEP)
        if model_id and voice:
            return model_id, voice
    return None


def _resolve_combined_tts_value(
    tts_info: dict[str, _TTSModelInfo],
    user_input: dict[str, Any] | None,
    saved_options: dict[str, Any],
) -> str:
    """Return the 'model → voice' string that should be pre-selected.

    Priority:
    1. A valid combined value already present in the current form submission.
    2. Reconstructed from the previously saved CONF_TTS_MODEL + CONF_TTS_VOICE.
    3. The default voice of the recommended (or first available) model.
    """
    # 1. Current submission
    if user_input is not None:
        submitted = user_input.get(_CONF_TTS_MODEL_VOICE)
        if isinstance(submitted, str):
            parsed = _parse_combined_tts_value(submitted)
            if parsed:
                model_id, voice = parsed
                info = tts_info.get(model_id)
                if info and voice in info.voices:
                    return submitted

    # 2. Saved options
    saved_model = saved_options.get(CONF_TTS_MODEL)
    saved_voice = saved_options.get(CONF_TTS_VOICE)
    if isinstance(saved_model, str) and isinstance(saved_voice, str):
        info = tts_info.get(saved_model)
        if info and saved_voice in info.voices:
            return f"{saved_model}{_TTS_MV_SEP}{saved_voice}"

    # 3. Fallback: first voice of recommended (or first available) model
    for candidate in [RECOMMENDED_TTS_MODEL] + sorted(tts_info):
        info = tts_info.get(candidate)
        if info and info.voices:
            voice = info.default_voice if info.default_voice else info.voices[0]
            return f"{candidate}{_TTS_MV_SEP}{voice}"

    return f"{RECOMMENDED_TTS_MODEL}{_TTS_MV_SEP}{RECOMMENDED_TTS_VOICE}"


# ---------------------------------------------------------------------------
# Options flow
# ---------------------------------------------------------------------------

class VeniceAIOptionsFlow(_OptionsFlowBase):
    """Options flow for Venice AI.

    Subclasses ``OptionsFlowWithReload`` (HA ≥ 2024.1) so the integration is
    automatically reloaded once the user saves their options.

    TTS model and voice are presented as a **single combined searchable
    dropdown** (``tts_model_voice``).  Each entry is formatted as
    ``"<model> → <voice>"`` so the user can type a model name to filter the
    list, then pick any voice for that model in one action.  The combined
    value is parsed on submit and stored as the separate ``tts_model`` and
    ``tts_voice`` keys that the rest of the integration already reads.

    This eliminates the previous two-step flow that required a re-render
    (or a second form page) whenever the TTS model changed, which was
    confusing and could result in lost settings if the user closed the
    second window.
    """

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        super().__init__()
        self._config_entry = config_entry

    @property
    def config_entry(self) -> ConfigEntry:
        """Return the config entry."""
        return self._config_entry

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
        immediately visible.

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

        async with AsyncVeniceAIClient(
            api_key=api_key,
            http_client=get_async_client(self.hass),
        ) as client:
            try:
                _LOGGER.debug("Fetching text models for options flow")
                text_resp = await client.models.list(model_type="text")
                if isinstance(text_resp, list):
                    chat_options = [
                        SelectOptionDict(label=m.get("id", "Unknown"), value=m.get("id", ""))
                        for m in text_resp
                        if isinstance(m, dict) and m.get("id")
                    ]
                    _LOGGER.debug("Found %d text models", len(chat_options))
            except AuthenticationError:
                _LOGGER.error("Authentication error fetching text models")
                errors["base"] = "invalid_auth"
            except VeniceAIError as err:
                _LOGGER.warning("Failed to fetch text models: %s", err)
            except Exception:
                _LOGGER.exception("Unexpected error fetching text models")

            try:
                _LOGGER.debug("Fetching TTS models for options flow")
                tts_resp = await client.models.list(model_type="tts")
                if isinstance(tts_resp, list):
                    tts_info = _extract_tts_model_info(tts_resp)
                    _LOGGER.debug("Found %d TTS models with voices", len(tts_info))
            except AuthenticationError:
                _LOGGER.error("Authentication error fetching TTS models")
                errors["base"] = "invalid_auth"
            except VeniceAIError as err:
                _LOGGER.warning("Failed to fetch TTS models: %s", err)
            except Exception:
                _LOGGER.exception("Unexpected error fetching TTS models")

            try:
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
                _LOGGER.error("Authentication error fetching ASR models")
                errors["base"] = "invalid_auth"
            except VeniceAIError as err:
                _LOGGER.warning("Failed to fetch ASR models: %s", err)
            except Exception:
                _LOGGER.exception("Unexpected error fetching ASR models")

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
        """Validate numeric option ranges and return per-field error keys."""
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

    def _build_options_schema(
        self,
        chat_options: list[SelectOptionDict],
        combined_tts_options: list[SelectOptionDict],
        stt_options: list[SelectOptionDict],
        llm_api_options: list[SelectOptionDict] | None = None,
    ) -> vol.Schema:
        """Build the full options schema.

        TTS model and voice are presented as a single combined searchable
        dropdown (``tts_model_voice``).  The user types a model name to
        filter, then picks a 'model → voice' entry.  Both are stored
        atomically on submit — no re-render handshake required.
        """
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
                # Single combined TTS model + voice selector.
                # Entries look like "kokoro → af_heart".  The dropdown is
                # searchable — type a model name to filter to its voices.
                vol.Optional(_CONF_TTS_MODEL_VOICE): SelectSelector(
                    SelectSelectorConfig(
                        options=combined_tts_options,
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

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Single-step options form: all settings including TTS voice selection.

        TTS model and voice are presented as one combined searchable dropdown
        (``tts_model_voice``).  Entries are formatted as ``"<model> → <voice>"``.
        The user types a model name to filter to its voices, picks one entry,
        and both model and voice are set simultaneously on submit — no
        re-render, no second page, no abandonment risk.
        """
        errors: dict[str, str] = {}

        # Always fetch live so newly added models/voices are immediately visible.
        chat_options, tts_info, stt_options, fetch_errors = await self._fetch_model_metadata()
        llm_api_options = await self._fetch_llm_api_options()
        combined_tts_options = _build_combined_tts_options(tts_info)

        if fetch_errors:
            errors.update(fetch_errors)

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
                # Parse the combined 'model → voice' selector value.
                combined_value = user_input.get(_CONF_TTS_MODEL_VOICE)
                parsed = (
                    _parse_combined_tts_value(combined_value)
                    if isinstance(combined_value, str)
                    else None
                )

                if parsed is None:
                    errors[_CONF_TTS_MODEL_VOICE] = "invalid_tts_voice_for_model"
                    _LOGGER.warning(
                        "Could not parse combined TTS value '%s'", combined_value
                    )
                else:
                    tts_model, tts_voice = parsed
                    # Build the final options dict: remove the form-only combined
                    # key and replace with the two separate stored keys.
                    final_options = {
                        k: v
                        for k, v in user_input.items()
                        if k != _CONF_TTS_MODEL_VOICE
                    }
                    final_options[CONF_TTS_MODEL] = tts_model
                    final_options[CONF_TTS_VOICE] = tts_voice
                    _LOGGER.debug(
                        "Options saved: TTS model='%s', voice='%s'",
                        tts_model,
                        tts_voice,
                    )
                    return self.async_create_entry(title="", data=final_options)

        # --- Build schema and suggested values for this render ---
        options_schema = self._build_options_schema(
            chat_options,
            combined_tts_options,
            stt_options,
            llm_api_options,
        )

        # Layer: defaults → saved options → current submission.
        # The combined TTS field is always resolved last so it accurately
        # reflects either the user's current selection or the saved state.
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
            CONF_TTS_RESPONSE_FORMAT: RECOMMENDED_TTS_RESPONSE_FORMAT,
            CONF_TTS_SPEED: RECOMMENDED_TTS_SPEED,
            CONF_STT_MODEL: RECOMMENDED_STT_MODEL,
            CONF_STT_RESPONSE_FORMAT: RECOMMENDED_STT_RESPONSE_FORMAT,
            CONF_STT_TIMESTAMPS: RECOMMENDED_STT_TIMESTAMPS,
            CONF_REQUEST_TIMEOUT: RECOMMENDED_REQUEST_TIMEOUT,
        }
        # Saved options may contain CONF_TTS_MODEL / CONF_TTS_VOICE separately —
        # those keys are not in the schema and are harmlessly ignored by
        # add_suggested_values_to_schema, but they are used by
        # _resolve_combined_tts_value below.
        suggested_values.update(self.config_entry.options)
        if user_input is not None:
            suggested_values.update(user_input)

        # Always resolve the combined TTS field last so it reflects the correct
        # pre-selection regardless of what was merged above.
        suggested_values[_CONF_TTS_MODEL_VOICE] = _resolve_combined_tts_value(
            tts_info, user_input, self.config_entry.options
        )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                options_schema, suggested_values
            ),
            errors=errors,
        )
