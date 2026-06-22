"""Unit tests for the TTS helper logic in config_flow.py.

Home Assistant is not available in this test environment, so the pure
helpers are extracted from ``custom_components/venice_ai/config_flow.py``
with AST and executed against the real constants from ``const.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FLOW_PATH = ROOT / "custom_components" / "venice_ai" / "config_flow.py"
CONST_PATH = ROOT / "custom_components" / "venice_ai" / "const.py"


def _load_const_module() -> types.ModuleType:
    """Load const.py directly to avoid importing the HA-dependent package."""
    spec = importlib.util.spec_from_file_location("venice_ai_const", CONST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create module spec for const.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONST = _load_const_module()
CONF_TTS_MODEL = CONST.CONF_TTS_MODEL
CONF_TTS_VOICE = CONST.CONF_TTS_VOICE
RECOMMENDED_TTS_MODEL = CONST.RECOMMENDED_TTS_MODEL
RECOMMENDED_TTS_VOICE = CONST.RECOMMENDED_TTS_VOICE


def _stub_select_option_dict(value: str, label: str) -> dict[str, str]:
    """Minimal stand-in for homeassistant.helpers.selector.SelectOptionDict."""
    return {"value": value, "label": label}


def _load_config_flow_helpers() -> types.ModuleType:
    """Parse config_flow.py and return a module with the extracted helpers."""
    source = CONFIG_FLOW_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    names_to_extract = {
        "_TTSModelInfo",
        "_parse_combined_tts_value",
        "_resolve_combined_tts_value",
        "_extract_tts_model_info",
        "_build_combined_tts_options",
    }
    extracted_nodes: list[ast.ClassDef | ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names_to_extract:
            extracted_nodes.append(node)

    if len(extracted_nodes) != len(names_to_extract):
        missing = names_to_extract - {n.name for n in extracted_nodes}
        raise RuntimeError(f"Failed to extract helpers: {missing}")

    module = types.ModuleType("config_flow_helpers")
    module.__dict__["SelectOptionDict"] = _stub_select_option_dict
    module.__dict__["CONF_TTS_MODEL"] = CONF_TTS_MODEL
    module.__dict__["CONF_TTS_VOICE"] = CONF_TTS_VOICE
    module.__dict__["RECOMMENDED_TTS_MODEL"] = RECOMMENDED_TTS_MODEL
    module.__dict__["RECOMMENDED_TTS_VOICE"] = RECOMMENDED_TTS_VOICE
    module.__dict__["_CONF_TTS_MODEL_VOICE"] = "tts_model_voice"
    module.__dict__["_TTS_MV_SEP"] = " → "
    module.__dict__["Any"] = Any
    module.__dict__["dict"] = dict
    module.__dict__["list"] = list
    module.__dict__["isinstance"] = isinstance
    module.__dict__["str"] = str
    module.__dict__["next"] = next
    module.__dict__["iter"] = iter

    for node in extracted_nodes:
        code = compile(ast.Module(body=[node], type_ignores=[]), CONFIG_FLOW_PATH.name, "exec")
        exec(code, module.__dict__)  # noqa: S102

    return module


@pytest.fixture(scope="module")
def helpers() -> types.ModuleType:
    return _load_config_flow_helpers()


class TestTTSModelInfoExtraction:
    """Tests for ``_extract_tts_model_info``."""

    def test_model_spec_voices_extracted(self, helpers: types.ModuleType) -> None:
        info = helpers._extract_tts_model_info(
            [
                {
                    "id": "tts-kokoro",
                    "model_spec": {
                        "voices": ["bm_daniel", "am_liam"],
                    },
                }
            ]
        )
        assert set(info.keys()) == {"tts-kokoro"}
        assert info["tts-kokoro"].voices == ["bm_daniel", "am_liam"]
        assert info["tts-kokoro"].default_voice == "bm_daniel"

    def test_explicit_default_voice(self, helpers: types.ModuleType) -> None:
        info = helpers._extract_tts_model_info(
            [
                {
                    "id": "tts-kokoro",
                    "default_voice": "am_liam",
                    "model_spec": {"voices": ["bm_daniel", "am_liam"]},
                }
            ]
        )
        assert info["tts-kokoro"].default_voice == "am_liam"

    def test_legacy_voice_models_fallback(self, helpers: types.ModuleType) -> None:
        info = helpers._extract_tts_model_info(
            [
                {
                    "id": "tts-legacy",
                    "voice_models": ["voice_a", "voice_b"],
                }
            ]
        )
        assert info["tts-legacy"].voices == ["voice_a", "voice_b"]
        assert info["tts-legacy"].default_voice == "voice_a"

    def test_models_without_voices_are_ignored(self, helpers: types.ModuleType) -> None:
        info = helpers._extract_tts_model_info(
            [
                {"id": "tts-empty", "model_spec": {"voices": []}},
                {"id": "tts-missing"},
            ]
        )
        assert info == {}


class TestParseCombinedTTSValue:
    """Tests for ``_parse_combined_tts_value``."""

    def test_parses_valid_value(self, helpers: types.ModuleType) -> None:
        assert helpers._parse_combined_tts_value("tts-kokoro → bm_daniel") == (
            "tts-kokoro",
            "bm_daniel",
        )

    def test_returns_none_for_invalid_value(self, helpers: types.ModuleType) -> None:
        assert helpers._parse_combined_tts_value("no-separator") is None
        assert helpers._parse_combined_tts_value(" → voice") is None
        assert helpers._parse_combined_tts_value("model → ") is None


class TestResolveCombinedTTSValue:
    """Tests for ``_resolve_combined_tts_value``."""

    @pytest.fixture
    def tts_info(self, helpers: types.ModuleType) -> dict[str, Any]:
        return {
            "tts-kokoro": helpers._TTSModelInfo(
                "tts-kokoro", ["bm_daniel", "am_liam"], "bm_daniel"
            ),
            "tts-eleven": helpers._TTSModelInfo("tts-eleven", ["rachel"], "rachel"),
        }

    def test_submitted_value_takes_priority(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_combined_tts_value(
            tts_info,
            {"tts_model_voice": "tts-eleven → rachel"},
            {CONF_TTS_MODEL: "tts-kokoro", CONF_TTS_VOICE: "bm_daniel"},
        )
        assert result == "tts-eleven → rachel"

    def test_saved_options_used_when_no_submission(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_combined_tts_value(
            tts_info,
            None,
            {CONF_TTS_MODEL: "tts-eleven", CONF_TTS_VOICE: "rachel"},
        )
        assert result == "tts-eleven → rachel"

    def test_recommended_default_fallback(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_combined_tts_value(tts_info, None, {})
        assert result == f"{RECOMMENDED_TTS_MODEL} → {RECOMMENDED_TTS_VOICE}"

    def test_first_model_when_recommended_missing(
        self, helpers: types.ModuleType
    ) -> None:
        info = {"tts-other": helpers._TTSModelInfo("tts-other", ["v1"], "v1")}
        result = helpers._resolve_combined_tts_value(info, None, {})
        assert result == "tts-other → v1"

    def test_invalid_submission_falls_back_to_saved(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_combined_tts_value(
            tts_info,
            {"tts_model_voice": "invalid-value"},
            {CONF_TTS_MODEL: "tts-kokoro", CONF_TTS_VOICE: "bm_daniel"},
        )
        assert result == "tts-kokoro → bm_daniel"


class TestBuildCombinedTTSOptions:
    """Tests for ``_build_combined_tts_options``."""

    def test_builds_options_for_all_models(self, helpers: types.ModuleType) -> None:
        tts_info = {
            "tts-kokoro": helpers._TTSModelInfo(
                "tts-kokoro", ["bm_daniel", "am_liam"], "bm_daniel"
            ),
            "tts-eleven": helpers._TTSModelInfo("tts-eleven", ["rachel"], "rachel"),
        }
        options = helpers._build_combined_tts_options(tts_info)
        values = [o["value"] for o in options]
        assert values == [
            "tts-eleven → rachel",
            "tts-kokoro → bm_daniel",
            "tts-kokoro → am_liam",
        ]

    def test_empty_info_returns_recommended_fallback(
        self, helpers: types.ModuleType
    ) -> None:
        options = helpers._build_combined_tts_options({})
        assert len(options) == 1
        assert options[0]["value"] == f"{RECOMMENDED_TTS_MODEL} → {RECOMMENDED_TTS_VOICE}"
