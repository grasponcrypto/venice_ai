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
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import AsyncVeniceAIClient
from .const import (
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_TTS_RESPONSE_FORMAT,
    RECOMMENDED_TTS_SPEED,
    RECOMMENDED_TTS_VOICE,
    VENICE_TTS_VOICES,
)


_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI TTS platform."""
    client: AsyncVeniceAIClient = config_entry.runtime_data
    async_add_entities([VeniceAITTS(client, config_entry)])


class VeniceAITTS(TextToSpeechEntity):
    """Venice AI TTS entity."""

    def __init__(
        self, client: AsyncVeniceAIClient, config_entry: ConfigEntry
    ) -> None:
        """Initialize TTS entity."""
        self._client = client
        self._config_entry = config_entry
        self._name = "Venice AI TTS"
        self._attr_unique_id = f"{config_entry.entry_id}_tts"
        self._attr_device_info = {
            "identifiers": {(config_entry.domain, config_entry.entry_id)},
            "name": "Venice AI",
            "manufacturer": "Venice AI",
        }

    @property
    def name(self) -> str:
        """Friendly name for entity listing."""
        return self._name

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return ["en"]

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    def supported_options(self) -> list[str]:
        """Return list of supported options."""
        return [
            # Standard Home Assistant TTS options used by Voice Assistant UI.
            ATTR_VOICE,
            ATTR_AUDIO_OUTPUT,
            # Backward-compatible integration-specific options.
            "tts_voice",
            "tts_model",
            "tts_response_format",
            "tts_speed",
        ]

    @property
    def default_options(self) -> dict[str, Any]:
        """Return default options."""
        return {
            ATTR_AUDIO_OUTPUT: RECOMMENDED_TTS_RESPONSE_FORMAT,
            "tts_voice": RECOMMENDED_TTS_VOICE,
            "tts_model": RECOMMENDED_TTS_MODEL,
            "tts_response_format": RECOMMENDED_TTS_RESPONSE_FORMAT,
            "tts_speed": RECOMMENDED_TTS_SPEED,
        }

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any] | None = None
    ) -> TtsAudioType:
        """Generate TTS audio."""
        if options is None:
            options = {}

        # Support both Home Assistant standard keys and existing Venice keys.
        voice = options.get(ATTR_VOICE) or options.get("tts_voice", RECOMMENDED_TTS_VOICE)
        model = options.get("tts_model", RECOMMENDED_TTS_MODEL)
        response_format = options.get(ATTR_AUDIO_OUTPUT) or options.get(
            "tts_response_format", RECOMMENDED_TTS_RESPONSE_FORMAT
        )
        speed = options.get("tts_speed", RECOMMENDED_TTS_SPEED)

        _LOGGER.debug("Generating TTS for message: %s", message)
        _LOGGER.debug(
            "TTS options: voice=%s, model=%s, format=%s, speed=%s",
            voice, model, response_format, speed
        )

        # Generate audio using Venice AI
        _LOGGER.debug("Calling Venice AI speech API")
        audio_data = await self._client.speech.generate(
            text=message,
            voice=voice,
            model=model,
            response_format=response_format,
            speed=speed,
            streaming=False
        )
        _LOGGER.debug(
            "Received raw audio data from API: %d bytes",
            len(audio_data) if audio_data else 0
        )

        _LOGGER.debug(
            "Generated audio data: type=%s, length=%d bytes",
            type(audio_data), len(audio_data) if audio_data else 0
        )

        # Log audio data info for debugging
        if audio_data:
            null_count = audio_data.count(b'\x00')
            _LOGGER.debug(
                "Audio data: %d bytes, %d null bytes",
                len(audio_data), null_count
            )

            # Check audio format
            if len(audio_data) > 10:
                start_bytes = audio_data[:10]
                _LOGGER.debug("Audio starts with: %s", start_bytes.hex())
                if (audio_data.startswith(b'RIFF') and
                        b'WAVE' in audio_data[8:16]):
                    _LOGGER.debug("WAV format detected")
                elif audio_data.startswith(b'ID3'):
                    _LOGGER.debug("MP3 with ID3 tag detected")
                elif (audio_data.startswith(b'\xFF\xFB') or
                      audio_data.startswith(b'\xFF\xF3') or
                      audio_data.startswith(b'\xFF\xF2')):
                    _LOGGER.debug("MP3 with MPEG frame detected")
                else:
                    _LOGGER.warning("Unknown audio format")

        # Return the audio data as raw bytes with format
        _LOGGER.debug("Returning TTS result: format=%s, type=%s",
                      response_format, type(audio_data))
        return (response_format, audio_data)

    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return available Venice voices for Home Assistant voice selection."""
        return [Voice(voice_id, voice_id) for voice_id in VENICE_TTS_VOICES]

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Stream audio using a single generated chunk."""
        message = "".join([chunk async for chunk in request.message_gen])
        options = dict(request.options or {})

        audio_format, audio_data = await self.async_get_tts_audio(
            message, request.language, options
        )
        if not audio_data or not audio_format:
            raise HomeAssistantError(f"No TTS from {self.entity_id} for '{message}'")

        async def gen() -> AsyncGenerator[bytes]:
            yield audio_data

        return TTSAudioResponse(audio_format, gen())
