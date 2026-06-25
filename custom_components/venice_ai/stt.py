"""Speech-to-Text provider for Venice AI."""
from __future__ import annotations

import asyncio
import datetime
import logging
import struct
import time
from collections.abc import AsyncIterable

from homeassistant.components import stt
from homeassistant.components.stt import (
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
    MAX_STT_BUFFER_SIZE,
    RECOMMENDED_STT_MODEL,
    RECOMMENDED_STT_RESPONSE_FORMAT,
    RECOMMENDED_STT_TIMESTAMPS,
)
from .client import AsyncVeniceAIClient, VeniceAIError

_LOGGER = logging.getLogger(__name__)

# Fix 2: Moved out of async_process_audio_stream to avoid re-creating on every call.
# Each tuple is (metadata_attr_name, property_name, human_label).
_STT_VALIDATION_ATTRS = [
    ("format", "supported_formats", "audio format"),
    ("codec", "supported_codecs", "audio codec"),
    ("bit_rate", "supported_bit_rates", "bit rate"),
    ("sample_rate", "supported_sample_rates", "sample rate"),
    ("channel", "supported_channels", "channel count"),
]


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
    async_add_entities([VeniceAISTT(entry)])


class VeniceAISTT(SpeechToTextEntity):
    """The Venice AI Speech-to-Text provider."""

    def __init__(
        self,
        entry: ConfigEntry,
    ) -> None:
        """Initialize Venice AI STT."""
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_stt"
        self._attr_name = entry.title
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="STT",
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
        # Validate metadata against declared supported formats
        for attr, prop, label in _STT_VALIDATION_ATTRS:
            supported = getattr(self, prop)
            if getattr(metadata, attr) not in supported:
                _LOGGER.error(
                    "Unsupported %s: %s. Only %s is supported.",
                    label, getattr(metadata, attr), supported,
                )
                return stt.SpeechResult("", stt.SpeechResultState.ERROR)

        try:
            _stt_start = time.monotonic()
            _LOGGER.debug(
                "[PERF-STT] [+0.000s] Audio stream received at %s — buffering audio",
                datetime.datetime.now().isoformat(timespec="milliseconds"),
            )

            # Read all data from the stream using bytearray for efficiency
            audio_data = bytearray()
            async for chunk in stream:
                audio_data.extend(chunk)
                if len(audio_data) > MAX_STT_BUFFER_SIZE:
                    _LOGGER.error(
                        "Audio buffer exceeded maximum size of %d bytes; aborting transcription",
                        MAX_STT_BUFFER_SIZE,
                    )
                    return stt.SpeechResult("", stt.SpeechResultState.ERROR)

            _buffered_t = time.monotonic() - _stt_start
            _LOGGER.debug(
                "[PERF-STT] [+%.3fs] Audio buffering complete — %d bytes received",
                _buffered_t,
                len(audio_data),
            )

            # Handle empty audio streams gracefully
            if len(audio_data) == 0:
                _LOGGER.warning("Received empty audio stream for transcription")
                return stt.SpeechResult("", stt.SpeechResultState.ERROR)

            # Read options dynamically so changes after setup take effect immediately
            model = self.entry.options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL)
            response_format = self.entry.options.get(
                CONF_STT_RESPONSE_FORMAT, RECOMMENDED_STT_RESPONSE_FORMAT
            )
            timestamps = self.entry.options.get(
                CONF_STT_TIMESTAMPS, RECOMMENDED_STT_TIMESTAMPS
            )

            _LOGGER.debug(
                "[PERF-STT] [+%.3fs] Processing audio (%d bytes) with model=%s, format=%s, timestamps=%s",
                time.monotonic() - _stt_start,
                len(audio_data),
                model,
                response_format,
                timestamps,
            )

            # Convert PCM data to WAV format since Venice AI expects proper WAV files
            wav_data = _pcm_to_wav(bytes(audio_data), sample_rate=16000, num_channels=1, bits_per_sample=16)
            _LOGGER.debug(
                "[PERF-STT] [+%.3fs] PCM→WAV conversion done (%d → %d bytes)",
                time.monotonic() - _stt_start,
                len(audio_data),
                len(wav_data),
            )

            client: AsyncVeniceAIClient = self.entry.runtime_data.client

            _LOGGER.debug(
                "[PERF-STT] [+%.3fs] Sending to Venice AI transcription API (model=%s)",
                time.monotonic() - _stt_start,
                model,
            )
            _api_start = time.monotonic()

            result = await client.transcriptions.create(
                audio_data=wav_data,
                model=model,
                response_format=response_format,
                timestamps=timestamps,
            )

            _api_elapsed = time.monotonic() - _api_start
            _total_elapsed = time.monotonic() - _stt_start
            text = result.get("text", "")
            _LOGGER.debug(
                "[PERF-STT] [+%.3fs] Transcription received in %.3fs — %d chars: %r",
                _total_elapsed,
                _api_elapsed,
                len(text),
                text[:100] if text else "<empty>",
            )

            return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)

        except VeniceAIError as err:
            _LOGGER.error("Venice AI transcription error: %s", err)
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _LOGGER.exception("Unexpected error during transcription: %s", err)
            return stt.SpeechResult("", stt.SpeechResultState.ERROR)
