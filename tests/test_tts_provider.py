import importlib
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from hass_stubs import install_homeassistant_stubs

install_homeassistant_stubs()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)


def _install_venice_namespace_package() -> None:
    """Avoid importing integration __init__.py during unit tests."""
    if "custom_components" not in sys.modules:
        pkg = types.ModuleType("custom_components")
        pkg.__path__ = [os.path.join(BASE_DIR, "custom_components")]
        sys.modules["custom_components"] = pkg

    if "custom_components.venice_ai" not in sys.modules:
        pkg = types.ModuleType("custom_components.venice_ai")
        pkg.__path__ = [os.path.join(BASE_DIR, "custom_components", "venice_ai")]
        sys.modules["custom_components.venice_ai"] = pkg


_install_venice_namespace_package()
tts_module = importlib.import_module("custom_components.venice_ai.tts")
const_module = importlib.import_module("custom_components.venice_ai.const")


class DummySpeech:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return b"audio-bytes"


class DummyClient:
    def __init__(self):
        self.speech = DummySpeech()


class DummyEntry:
    def __init__(self):
        self.entry_id = "entry-1"
        self.domain = "venice_ai"


@pytest.mark.asyncio
async def test_supported_voices_exposed_for_voice_assistant_ui():
    provider = tts_module.VeniceAITTS(DummyClient(), DummyEntry())
    voices = provider.async_get_supported_voices("en")

    assert voices is not None
    assert len(voices) == len(const_module.VENICE_TTS_VOICES)
    assert voices[0].voice_id == const_module.VENICE_TTS_VOICES[0]


@pytest.mark.asyncio
async def test_tts_audio_prefers_standard_ha_options():
    client = DummyClient()
    provider = tts_module.VeniceAITTS(client, DummyEntry())
    fmt, data = await provider.async_get_tts_audio(
        "hello",
        "en",
        {
            "voice": "af_alloy",
            "audio_output": "mp3",
            "tts_model": "tts-kokoro",
            "tts_speed": 1.4,
        },
    )

    assert fmt == "mp3"
    assert data == b"audio-bytes"
    assert client.speech.calls[0]["voice"] == "af_alloy"
    assert client.speech.calls[0]["response_format"] == "mp3"
    assert client.speech.calls[0]["model"] == "tts-kokoro"
    assert client.speech.calls[0]["speed"] == 1.4


@pytest.mark.asyncio
async def test_tts_audio_supports_legacy_venice_option_keys():
    client = DummyClient()
    provider = tts_module.VeniceAITTS(client, DummyEntry())
    fmt, _ = await provider.async_get_tts_audio(
        "hello",
        "en",
        {
            "tts_voice": "bm_daniel",
            "tts_response_format": "wav",
            "tts_model": "tts-kokoro",
            "tts_speed": 1.0,
        },
    )

    assert fmt == "wav"
    assert client.speech.calls[0]["voice"] == "bm_daniel"
    assert client.speech.calls[0]["response_format"] == "wav"

