"""services.yaml must describe the action fields the UI renders."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / "custom_components" / "venice_ai" / "services.yaml"


def test_generate_image_fields_are_declared() -> None:
    text = SERVICES.read_text(encoding="utf-8")
    assert "generate_image:" in text
    for field in (
        "config_entry:",
        "prompt:",
        "model:",
        "aspect_ratio:",
        "resolution:",
        "width:",
        "height:",
    ):
        assert field in text
    assert "size:" not in text
    assert "quality:" not in text
    assert "style:" not in text
