"""STT state is a timestamp. The model id is an attribute, on TTS too."""

from __future__ import annotations

import datetime
import sys
import types
from pathlib import Path

from .conftest import load_component_module

ROOT = Path(__file__).resolve().parents[1]
STT_PATH = ROOT / "custom_components" / "venice_ai" / "stt.py"
TTS_PATH = ROOT / "custom_components" / "venice_ai" / "tts.py"
INIT_PATH = ROOT / "custom_components" / "venice_ai" / "__init__.py"


def _ensure_module(name: str) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module = types.ModuleType(name)
    if name.count(".") == 0 or name.endswith(".components") or name.endswith(".helpers"):
        module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        parent = _ensure_module(parent_name)
        setattr(parent, child, module)
    return module


def _install_stt_stub() -> None:
    """Stub the Home Assistant names stt.py and tts.py import."""
    stt_mod = _ensure_module("homeassistant.components.stt")

    class SpeechToTextEntity:
        """2026.10 state: the name-mangled last-processed timestamp."""

        def __init__(self) -> None:
            self._SpeechToTextEntity__last_processed: str | None = None

        @property
        def state(self) -> str | None:
            return self._SpeechToTextEntity__last_processed

        async def async_internal_added_to_hass(self) -> None:
            return None

    stt_mod.SpeechToTextEntity = SpeechToTextEntity  # type: ignore[attr-defined]
    components = _ensure_module("homeassistant.components")
    components.stt = stt_mod  # type: ignore[attr-defined]

    tts_mod = _ensure_module("homeassistant.components.tts")
    tts_mod.ATTR_AUDIO_OUTPUT = "audio_output"  # type: ignore[attr-defined]
    tts_mod.ATTR_VOICE = "voice"  # type: ignore[attr-defined]
    tts_mod.TTSAudioRequest = object  # type: ignore[attr-defined]
    tts_mod.TTSAudioResponse = object  # type: ignore[attr-defined]
    tts_mod.TextToSpeechEntity = object  # type: ignore[attr-defined]
    tts_mod.TtsAudioType = object  # type: ignore[attr-defined]
    tts_mod.Voice = object  # type: ignore[attr-defined]

    _ensure_module("homeassistant.config_entries").ConfigEntry = object  # type: ignore[attr-defined]
    _ensure_module("homeassistant.core").HomeAssistant = object  # type: ignore[attr-defined]
    _ensure_module("homeassistant.exceptions").HomeAssistantError = Exception  # type: ignore[attr-defined]

    registry = _ensure_module("homeassistant.helpers.device_registry")

    class DeviceInfo:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class DeviceEntryType:
        SERVICE = "service"

    registry.DeviceInfo = DeviceInfo  # type: ignore[attr-defined]
    registry.DeviceEntryType = DeviceEntryType  # type: ignore[attr-defined]
    _ensure_module("homeassistant.helpers.entity_platform").AddEntitiesCallback = object  # type: ignore[attr-defined]

    dt_mod = _ensure_module("homeassistant.util.dt")

    def utcnow() -> datetime.datetime:
        return datetime.datetime(2026, 11, 3, 12, 0, tzinfo=datetime.timezone.utc)

    dt_mod.utcnow = utcnow  # type: ignore[attr-defined]


def test_stt_does_not_override_timestamp_state() -> None:
    text = STT_PATH.read_text(encoding="utf-8")
    assert "def state(" not in text
    assert "def extra_state_attributes(" in text
    assert "_SpeechToTextEntity__last_processed" in text
    assert "async_internal_added_to_hass" in text
    assert "Invalid timestamp" in text


def test_tts_shows_model_without_overriding_state() -> None:
    text = TTS_PATH.read_text(encoding="utf-8")
    assert "def state(" not in text
    assert "def extra_state_attributes(" in text
    assert "CONF_TTS_MODEL" in text


def test_ai_task_service_does_not_use_the_shadowed_name() -> None:
    text = INIT_PATH.read_text(encoding="utf-8")
    assert "import ai_task as ha_ai_task" in text
    assert "ha_ai_task.GenDataTask" in text
    assert "ai_task.GenDataTask" not in text.replace("ha_ai_task.GenDataTask", "")


def test_model_id_is_not_a_timestamp_and_assist_records_one() -> None:
    _install_stt_stub()
    stt = load_component_module("stt")

    assert stt.is_iso_timestamp("nvidia/parakeet-tdt-0.6b-v3") is False
    assert stt.is_iso_timestamp("unknown") is False
    assert stt.is_iso_timestamp("2026-11-03T12:00:00+00:00") is True

    entry = types.SimpleNamespace(
        entry_id="abc",
        title="Venice AI",
        options={"stt_model": "nvidia/parakeet-tdt-0.6b-v3"},
    )
    entity = stt.VeniceAISTT(entry)
    entity.hass = object()
    entity.entity_id = "stt.venice_ai_stt"
    written: list[str | None] = []
    entity.async_write_ha_state = lambda: written.append(entity.state)

    entity._SpeechToTextEntity__last_processed = "nvidia/parakeet-tdt-0.6b-v3"
    entity._drop_non_timestamp_state()
    assert entity.state is None
    assert written == [None]

    entity._record_processed_time()
    assert entity.state == "2026-11-03T12:00:00+00:00"
    assert written[-1] == "2026-11-03T12:00:00+00:00"

    written.clear()
    entity._drop_non_timestamp_state()
    assert written == []
    assert entity.extra_state_attributes == {"model": "nvidia/parakeet-tdt-0.6b-v3"}


def test_tts_attribute_is_the_configured_model() -> None:
    _install_stt_stub()
    tts = load_component_module("tts")
    entry = types.SimpleNamespace(
        entry_id="abc",
        title="Venice AI",
        options={"tts_model": "kokoro"},
        runtime_data=types.SimpleNamespace(client=object(), coordinator=None),
    )
    entity = tts.VeniceAITTS(entry)
    assert entity.extra_state_attributes == {"model": "kokoro"}

