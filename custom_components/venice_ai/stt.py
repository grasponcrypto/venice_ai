"""Speech-to-Text provider for Venice AI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from homeassistant.components.stt import (
    Provider,
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


class VeniceAISTT(Provider):
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
        return ["wav", "wave", "flac"]

    @property
    def supported_sample_rates(self) -> list[int]:
        """Return list of supported sample rates in Hz."""
        return [16000]

    @property
    def supported_codecs(self) -> list[str]:
        """Return list of supported audio codecs."""
        return ["pcm"]

    @property
    def supported_bit_rates(self) -> list[int]:
        """Return list of supported bit rates in bps."""
        return [16000]

    @property
    def supported_channels(self) -> list[int]:
        """Return list of supported channel counts."""
        return [1]

    async def async_process_audio_stream(
        self,
        stream,
        metadata: dict[str, Any] | None = None,
    ) -> SpeechResult:
        """Process an audio stream to text."""
        from .client import AsyncVeniceAIClient, VeniceAIError

        try:
            # Read all data from the stream
            audio_data = b""
            async for chunk in stream:
                audio_data += chunk

            LOGGER.debug(
                "Processing audio stream (%d bytes) with model=%s, format=%s, timestamps=%s",
                len(audio_data),
                self._model,
                self._response_format,
                self._timestamps,
            )

            client: AsyncVeniceAIClient = self._entry.runtime_data

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
