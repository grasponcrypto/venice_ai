"""STT must not put the model id in entity state.

Home Assistant 2026.10 formats speech-to-text state as a timestamp. A model
id there is what the UI calls Invalid timestamp.
"""

from __future__ import annotations

from pathlib import Path

STT = Path(__file__).resolve().parents[1] / "custom_components" / "venice_ai" / "stt.py"


def test_stt_does_not_override_timestamp_state() -> None:
    text = STT.read_text(encoding="utf-8")
    assert "def state(" not in text
    assert "def extra_state_attributes(" in text
    assert "Invalid timestamp" in text
