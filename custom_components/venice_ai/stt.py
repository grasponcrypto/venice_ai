"""Speech-to-Text provider for Venice AI."""
from __future__ import annotations

import logging
import struct
from collections.abc import AsyncIterable
from typing import Any

from homeassistant.components import stt
from homeassistant.components.stt import (
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_STT_MODEL,
    CONF_STT_RESPONSE_FORMAT,
    CONF_STT_TIMESTAMPS,
    DOMAIN,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_STT_RESPONSE_FORMAT,
    RECOMMENDED_STT_TIMESTAMPS,
)

_LOGGER = logging.getLogger(__name__)


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Convert raw PCM data to WAV format."""
    # Calculate sizes
    subchunk2_size = len(pcm_data)
    chunk_size = 36 + subchunk2_size
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # WAV header (44 bytes)
    wav_header = struct.pack(
        '<4sL4s4sLHHLLHH4sL',
        b'RIFF',  # ChunkID
        chunk_size,  # ChunkSize
        b'WAVE',  # Format
        b'fmt ',  # Subchunk1ID
        16,  # Subchunk1Size (PCM)
        1,  # AudioFormat (PCM)
        num_channels,  # NumChannels
        sample_rate,  # SampleRate
        byte_rate,  # ByteRate
        block_align,  # BlockAlign
        bits_per_sample,  # BitsPerSample
        b'data',  # Subchunk2ID
        subchunk2_size,  # Subchunk2Size
    )

    return wav_header + pcm_data


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI STT entity."""
    async_add_entities([
        VeniceAISTT(
            entry,
            entry.options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
            entry.options.get(
                CONF_STT_RESPONSE_FORMAT, RECOMMENDED_STT_RESPONSE_FORMAT
            ),
            entry.options.get(CONF_STT_TIMESTAMPS, RECOMMENDED_STT_TIMESTAMPS),
        )
    ])


class VeniceAISTT(SpeechToTextEntity):
    """The Venice AI Speech-to-Text provider."""

    def __init__(
        self,
        entry: ConfigEntry,
        model: str,
        response_format: str,
        timestamps: bool,
    ) -> None:
        """Initialize Venice AI STT."""
        self.entry = entry
        self._model = model
        self._response_format = response_format
        self._timestamps = timestamps
        self._attr_unique_id = f"{entry.entry_id}_stt"
        self._attr_name = entry.title
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="Venice AI STT",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        # Venice AI parakeet model supports these languages
        return ["en", "zh", "fr", "hi", "it", "ja", "pl", "es"]

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return list of supported audio formats."""
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return list of supported audio codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return list of supported bit rates in bps."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return list of supported sample rates in Hz."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return list of supported channel counts."""
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self,
        metadata: stt.SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> stt.SpeechResult:
        """Process an audio stream to text.

        Note: This implementation buffers the entire stream in memory,
        converts it to WAV format, and sends it as a single request.
        Venice AI does not currently support chunked streaming uploads
        for transcriptions.
        """
        from .client import AsyncVeniceAIClient, VeniceAIError

        # Validate metadata against declared supported formats
        if metadata.format not in self.supported_formats:
            _LOGGER.error(
                "Unsupported audio format: %s. Only %s is supported.",
                metadata.format, self.supported_formats,
            )
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
        if metadata.codec not in self.supported_codecs:
            _LOGGER.error(
                "Unsupported audio codec: %s. Only %s is supported.",
                metadata.codec, self.supported_codecs,
            )
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
        if metadata.bit_rate not in self.supported_bit_rates:
            _LOGGER.error(
                "Unsupported bit rate: %s. Only %s is supported.",
                metadata.bit_rate, self.supported_bit_rates,
            )
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
        if metadata.sample_rate not in self.supported_sample_rates:
            _LOGGER.error(
                "Unsupported sample rate: %s. Only %s is supported.",
                metadata.sample_rate, self.supported_sample_rates,
            )
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
        if metadata.channel not in self.supported_channels:
            _LOGGER.error(
                "Unsupported channel count: %s. Only %s is supported.",
                metadata.channel, self.supported_channels,
            )
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)

        try:
            # Read all data from the stream using bytearray for efficiency
            audio_data = bytearray()
            async for chunk in stream:
                audio_data.extend(chunk)

            # Handle empty audio streams gracefully
            if len(audio_data) == 0:
                _LOGGER.warning("Received empty audio stream for transcription")
                return stt.SpeechResult("", stt.SpeechResultState.ERROR)

            _LOGGER.debug(
                "Processing audio stream (%d bytes) with model=%s, format=%s, timestamps=%s",
                len(audio_data),
                self._model,
                self._response_format,
                self._timestamps,
            )

            # Convert PCM data to WAV format since Venice AI expects proper WAV files
            wav_data = _pcm_to_wav(bytes(audio_data), sample_rate=16000, num_channels=1, bits_per_sample=16)
            _LOGGER.debug("Converted PCM to WAV (%d bytes -> %d bytes)", len(audio_data), len(wav_data))

            client: AsyncVeniceAIClient = self.entry.runtime_data.client

            result = await client.transcriptions.create(
                audio_data=wav_data,
                model=self._model,
                response_format=self._response_format,
                timestamps=self._timestamps,
            )

            text = result.get("text", "")
            _LOGGER.debug("Transcription result: %s", text)

            return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)

        except VeniceAIError as err:
            _LOGGER.error("Venice AI transcription error: %s", err)
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
        except Exception as err:
            _LOGGER.exception("Unexpected error during transcription: %s", err)
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
