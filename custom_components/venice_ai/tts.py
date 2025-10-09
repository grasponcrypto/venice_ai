"""Venice AI TTS platform."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import AsyncVeniceAIClient, VeniceAIError

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

    def __init__(self, client: AsyncVeniceAIClient, config_entry: ConfigEntry) -> None:
        """Initialize TTS entity."""
        self._client = client
        self._config_entry = config_entry
        self._attr_unique_id = f"{config_entry.entry_id}_tts"
        self._attr_name = "Venice AI TTS"
        self._attr_device_info = {
            "identifiers": {(config_entry.domain, config_entry.entry_id)},
            "name": "Venice AI",
            "manufacturer": "Venice AI",
        }

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return ["en"]

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Generate TTS audio."""
        try:
            audio_data = await self._client.speech.generate(
                text=message,
                voice="bm_daniel"
            )
            return TtsAudioType(data=audio_data)
        except VeniceAIError as err:
            _LOGGER.error("Error generating TTS audio: %s", err)
            raise