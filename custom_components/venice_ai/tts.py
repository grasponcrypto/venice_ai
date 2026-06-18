"""Venice AI TTS platform."""
from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

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

from .client import AsyncVeniceAIClient
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

        _LOGGER.debug("Generating TTS for message: %s", message)
        _LOGGER.debug(
            "TTS options: voice=%s, model=%s, format=%s, speed=%s",
            voice, model, response_format, speed
        )

        audio_data = await self._client.speech.generate(
            text=message,
            voice=voice,
            model=model,
            audio_output=response_format,
            speed=speed,
        )
        _LOGGER.debug(
            "Received raw audio data from API: %d bytes",
            len(audio_data) if audio_data else 0
        )

        if not audio_data:
            raise HomeAssistantError("TTS generation returned empty audio")

        return (response_format, audio_data)

    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return available Venice voices for Home Assistant voice selection.

        Voices are pulled from the coordinator's cached audio-models data.
        When the coordinator hasn't finished its first update yet, an empty
        list is returned so the TTS entity degrades gracefully.
        """
        coordinator = getattr(self._config_entry.runtime_data, "coordinator", None)
        if coordinator is None or coordinator.data is None:
            return []

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

        _LOGGER.debug("Streaming TTS for message: %s", message)
        _LOGGER.debug(
            "Streaming TTS options: voice=%s, model=%s, format=%s, speed=%s",
            voice, model, response_format, speed
        )

        if not message:
            raise HomeAssistantError(f"No TTS message for {self.entity_id}")

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
