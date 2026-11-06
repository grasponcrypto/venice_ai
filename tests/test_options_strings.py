"""Guard the options-form error keys and the shared device model."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_COMPONENT = _ROOT / "custom_components" / "venice_ai"

_OPTIONS_ERROR_KEYS = {
    "invalid_auth",
    "cannot_connect",
    "unknown",
    "missing_api_key",
    "invalid_tts_voice_for_model",
    "invalid_llm_api",
    "max_tokens_out_of_range",
    "top_p_out_of_range",
    "temperature_out_of_range",
    "max_tool_iterations_out_of_range",
    "tts_speed_out_of_range",
    "prompt_must_be_string",
}


def test_options_error_keys_are_translated() -> None:
    strings = json.loads((_COMPONENT / "strings.json").read_text(encoding="utf-8"))
    english = json.loads(
        (_COMPONENT / "translations" / "en.json").read_text(encoding="utf-8")
    )
    assert _OPTIONS_ERROR_KEYS <= set(strings["options"]["error"])
    assert _OPTIONS_ERROR_KEYS <= set(english["options"]["error"])


def test_platforms_share_one_device_model() -> None:
    for name in ("conversation.py", "stt.py", "tts.py", "ai_task.py", "sensor.py"):
        text = (_COMPONENT / name).read_text(encoding="utf-8")
        assert "DEVICE_MODEL" in text
        assert 'model="Conversation"' not in text
        assert 'model="STT"' not in text
        assert 'model="TTS"' not in text
        assert 'model="AI Task"' not in text
