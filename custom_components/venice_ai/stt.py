"""Speech-to-Text provider for Venice AI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from homeassistant.components.stt import (
    SpeechToTextEntity,
    SpeechResult,
    SpeechResultState,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_STT_MODEL,
    CONF_STT_RESPONSE_FORMAT,
    CONF_STT_TIMESTAMPS,
    DOMAIN,
    LOGGER,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_STT_RESPONSE_FORMAT,
    RECOMMENDED_STT_TIMESTAMPS,
)

_LOGGER = logging.getLogger(__package__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI STT platforms."""
    async_add_entities(
        [
            VeniceAISTT(
                entry,
                entry.options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
                entry.options.get(
                    CONF_STT_RESPONSE_FORMAT, RECOMMENDED_STT_RESPONSE_FORMAT
                ),
                entry.options.get(CONF_STT_TIMESTAMPS, RECOMMENDED_STT_TIMESTAMPS),
            )
        ]
    )


class VeniceAISTT(SpeechToTextEntity):
    """The Venice AI Speech-to-Text integration."""

    def __init__(
        self,
        entry: ConfigEntry,
        model: str,
        response_format: str,
        timestamps: bool,
    ) -> None:
        """Initialize Venice AI STT."""
        self._entry = entry
        self._model = model
        self._response_format = response_format
        self._timestamps = timestamps
        self._attr_name = entry.title
        self._attr_unique_id = f"{entry.entry_id}_stt"

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        # Currently Venice AI supports English
        return ["en"]

    @property
    def supported_formats(self) -> list[str]:
        """Return list of supported audio formats."""
        return ["wav", "wave", "flac", "m4a", "aac", "mp4", "mp3"]

    async def async_process(
        self,
        audio_data: bytes | str,
        language: str | None = None,
    ) -> SpeechResult:
        """Process audio data to text."""
        from .client import AsyncVeniceAIClient, VeniceAIError

        client: AsyncVeniceAIClient = self._entry.runtime_data

        # If audio_data is a file path, read the file
        if isinstance(audio_data, str):
            try:
                audio_path = Path(audio_data)
                if not audio_path.exists():
                    LOGGER.error("Audio file not found: %s", audio_path)
                    return SpeechResult(
                        text="", result_state=SpeechResultState.ERROR
                    )
                audio_data = audio_path.read_bytes()
                LOGGER.debug("Read audio file: %s (%d bytes)", audio_path, len(audio_data))
            except Exception as err:
                LOGGER.error("Error reading audio file: %s", err)
                return SpeechResult(text="", result_state=SpeechResultState.ERROR)

        try:
            LOGGER.debug(
                "Starting transcription with model=%s, format=%s, timestamps=%s",
                self._model,
                self._response_format,
                self._timestamps,
            )

            result = await client.transcriptions.create(
                audio_data=audio_data,
                model=self._model,
                response_format=self._response_format,
                timestamps=self._timestamps,
            )

            text = result.get("text", "")
            LOGGER.debug("Transcription result: %s", text)

            return SpeechResult(text=text, result_state=SpeechResultState.SUCCESS)

        except VeniceAIError as err:
            LOGGER.error("Venice AI transcription error: %s", err)
            return SpeechResult(text="", result_state=SpeechResultState.ERROR)
        except Exception as err:
            LOGGER.exception("Unexpected error during transcription: %s", err)
            return SpeechResult(text="", result_state=SpeechResultState.ERROR)
