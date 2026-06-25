"""Venice AI TTS platform."""
from __future__ import annotations

import logging
import time
from typing import Any

from homeassistant.components.tts import (
    ATTR_AUDIO_OUTPUT,
    ATTR_VOICE,
    TTSAudioRequest,
    TTSAudioResponse,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_TTS_MODEL,
    CONF_TTS_RESPONSE_FORMAT,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    DOMAIN,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_RESPONSE_FORMAT,
    RECOMMENDED_TTS_SPEED,
    RECOMMENDED_TTS_VOICE,
)


_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI TTS platform."""
    async_add_entities([VeniceAITTS(config_entry)])


class VeniceAITTS(TextToSpeechEntity):
    """Venice AI TTS entity."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize TTS entity."""
        self._client = config_entry.runtime_data.client
        self._config_entry = config_entry
        self._name = "Venice AI TTS"
        self._attr_unique_id = f"{config_entry.entry_id}_tts"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            name=config_entry.title,
            manufacturer="Venice AI",
            model="TTS",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def name(self) -> str:
        """Friendly name for entity listing."""
        return self._name

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        # Venice AI kokoro TTS supports multiple languages via different voice sets
        return ["en", "zh", "fr", "hi", "it", "ja", "pl", "es"]

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    def supported_options(self) -> list[str]:
        """Return list of supported options."""
        return [
            ATTR_VOICE,
            ATTR_AUDIO_OUTPUT,
            CONF_TTS_MODEL,
            CONF_TTS_SPEED,
        ]

    @property
    def default_options(self) -> dict[str, Any]:
        """Return default options mapped from config entry."""
        options = self._config_entry.options
        return {
            ATTR_VOICE: options.get(CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE),
            ATTR_AUDIO_OUTPUT: options.get(CONF_TTS_RESPONSE_FORMAT, RECOMMENDED_TTS_RESPONSE_FORMAT),
            "tts_model": options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL),
            "tts_speed": options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED),
        }

    def _get_tts_option(
        self,
        options: dict[str, Any] | None,
        option_key: str,
        config_key: str,
        default: Any,
    ) -> Any:
        """Return a TTS option, falling back to config entry then recommended default."""
        if options is None:
            options = {}
        return options.get(
            option_key, self._config_entry.options.get(config_key, default)
        )

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any] | None = None
    ) -> TtsAudioType:
        """Generate TTS audio."""
        voice = self._get_tts_option(options, ATTR_VOICE, CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
        model = self._get_tts_option(options, "tts_model", CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL)
        response_format = self._get_tts_option(
            options, ATTR_AUDIO_OUTPUT, CONF_TTS_RESPONSE_FORMAT, RECOMMENDED_TTS_RESPONSE_FORMAT
        )
        speed = self._get_tts_option(options, "tts_speed", CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED)

        _tts_start = time.monotonic()
        _LOGGER.debug(
            "[PERF-TTS] [+0.000s] TTS request — text=%d chars, voice=%s, model=%s, format=%s, speed=%s",
            len(message),
            voice, model, response_format, speed,
        )

        _LOGGER.debug(
            "[PERF-TTS] [+%.3fs] Sending to Venice AI speech API",
            time.monotonic() - _tts_start,
        )
        _api_start = time.monotonic()

        audio_data = await self._client.speech.generate(
            text=message,
            voice=voice,
            model=model,
            audio_output=response_format,
            speed=speed,
        )

        _api_elapsed = time.monotonic() - _api_start
        _total_elapsed = time.monotonic() - _tts_start
        _LOGGER.debug(
            "[PERF-TTS] [+%.3fs] Audio received from Venice AI in %.3fs — %d bytes",
            _total_elapsed,
            _api_elapsed,
            len(audio_data) if audio_data else 0,
        )

        if not audio_data:
            raise HomeAssistantError("TTS generation returned empty audio")

        return (response_format, audio_data)

    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return available Venice voices for Home Assistant voice selection.

        Only voices belonging to the currently configured TTS model are
        returned.  Returning voices from all models at once would flood the
        pipeline UI with hundreds of entries that don't work with the active
        model.

        Falls back to all known voices if the configured model cannot be
        found in the coordinator cache (e.g. during first startup before the
        coordinator has refreshed).
        """
        coordinator = getattr(self._config_entry.runtime_data, "coordinator", None)
        if coordinator is None or coordinator.data is None:
            return []

        active_model = self._config_entry.options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL)

        # Find the active model in the coordinator's audio_models list and
        # extract its voices using the same dual-source logic as coordinator.py.
        audio_models: list[dict] = coordinator.data.get("audio_models", [])
        for model in audio_models:
            if not isinstance(model, dict):
                continue
            if model.get("id") != active_model:
                continue
            # Primary source: model_spec.voices
            raw_spec = model.get("model_spec")
            if isinstance(raw_spec, dict):
                spec_voices = raw_spec.get("voices")
                if isinstance(spec_voices, list):
                    voices = [v for v in spec_voices if isinstance(v, str) and v]
                    if voices:
                        return [Voice(v, v) for v in voices]
            # Fallback: legacy voice_models field
            legacy = model.get("voice_models", [])
            if isinstance(legacy, list):
                voices = [v for v in legacy if isinstance(v, str) and v]
                if voices:
                    return [Voice(v, v) for v in voices]

        # Active model not found in cache — fall back to all known voices so
        # the dropdown is never completely empty.
        voices_data: list[str] = coordinator.data.get("voices", [])
        return [Voice(voice_id, voice_id) for voice_id in voices_data]

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Stream audio using Venice AI's streaming TTS API."""
        message = "".join([chunk async for chunk in request.message_gen])
        options = dict(request.options or {})

        voice = self._get_tts_option(options, ATTR_VOICE, CONF_TTS_VOICE, RECOMMENDED_TTS_VOICE)
        model = self._get_tts_option(options, "tts_model", CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL)
        response_format = self._get_tts_option(
            options, ATTR_AUDIO_OUTPUT, CONF_TTS_RESPONSE_FORMAT, RECOMMENDED_TTS_RESPONSE_FORMAT
        )
        speed = self._get_tts_option(options, "tts_speed", CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED)

        _tts_start = time.monotonic()
        _LOGGER.debug(
            "[PERF-TTS] [+0.000s] Streaming TTS request — text=%d chars, voice=%s, model=%s, format=%s, speed=%s",
            len(message),
            voice, model, response_format, speed,
        )

        if not message:
            raise HomeAssistantError(f"No TTS message for {self.entity_id}")

        _LOGGER.debug(
            "[PERF-TTS] [+%.3fs] Opening streaming speech connection to Venice AI",
            time.monotonic() - _tts_start,
        )

        return TTSAudioResponse(
            response_format,
            self._client.speech.generate_streaming(
                text=message,
                voice=voice,
                model=model,
                audio_output=response_format,
                speed=speed,
            )
        )
