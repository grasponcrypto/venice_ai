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
        "_resolve_tts_model",
        "_resolve_tts_voice",
        "_extract_tts_model_info",
        "_build_voice_options",
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


class TestResolveTTSModel:
    """Tests for ``_resolve_tts_model``."""

    @pytest.fixture
    def tts_info(self, helpers: types.ModuleType) -> dict[str, Any]:
        return {
            "tts-kokoro": helpers._TTSModelInfo("tts-kokoro", ["bm_daniel"], "bm_daniel"),
            "tts-eleven": helpers._TTSModelInfo("tts-eleven", ["rachel"], "rachel"),
        }

    def test_submitted_model_takes_priority(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_tts_model(
            {CONF_TTS_MODEL: "tts-eleven"},
            {CONF_TTS_MODEL: "tts-kokoro"},
            tts_info,
        )
        assert result == "tts-eleven"

    def test_saved_model_used_when_no_submission(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_tts_model(None, {CONF_TTS_MODEL: "tts-eleven"}, tts_info)
        assert result == "tts-eleven"

    def test_recommended_default_fallback(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_tts_model(None, {}, tts_info)
        assert result == RECOMMENDED_TTS_MODEL

    def test_first_model_when_recommended_missing(
        self, helpers: types.ModuleType
    ) -> None:
        info = {"tts-other": helpers._TTSModelInfo("tts-other", ["v1"], "v1")}
        result = helpers._resolve_tts_model(None, {}, info)
        assert result == "tts-other"


class TestResolveTTSVoice:
    """Tests for ``_resolve_tts_voice``."""

    @pytest.fixture
    def tts_info(self, helpers: types.ModuleType) -> dict[str, Any]:
        return {
            "tts-kokoro": helpers._TTSModelInfo(
                "tts-kokoro", ["bm_daniel", "am_liam"], "bm_daniel"
            ),
            "tts-eleven": helpers._TTSModelInfo("tts-eleven", ["rachel"], "rachel"),
        }

    def test_submitted_voice_takes_priority(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_tts_voice(
            "tts-kokoro",
            tts_info,
            {CONF_TTS_VOICE: "am_liam"},
            {CONF_TTS_VOICE: "bm_daniel"},
        )
        assert result == "am_liam"

    def test_saved_voice_fallback(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_tts_voice(
            "tts-kokoro",
            tts_info,
            None,
            {CONF_TTS_VOICE: "am_liam"},
        )
        assert result == "am_liam"

    def test_defaults_to_model_default_voice(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_tts_voice("tts-kokoro", tts_info, None, {})
        assert result == "bm_daniel"

    def test_unknown_model_uses_recommended_voice(
        self, helpers: types.ModuleType, tts_info: dict[str, Any]
    ) -> None:
        result = helpers._resolve_tts_voice("tts-unknown", tts_info, None, {})
        assert result == RECOMMENDED_TTS_VOICE


class TestBuildVoiceOptions:
    """Tests for ``_build_voice_options``."""

    def test_builds_options_for_selected_model(
        self, helpers: types.ModuleType
    ) -> None:
        tts_info = {
            "tts-kokoro": helpers._TTSModelInfo(
                "tts-kokoro", ["bm_daniel", "am_liam"], "bm_daniel"
            ),
        }
        options = helpers._build_voice_options("tts-kokoro", tts_info)
        assert [o["value"] for o in options] == ["bm_daniel", "am_liam"]

    def test_unknown_model_returns_recommended_voice(
        self, helpers: types.ModuleType
    ) -> None:
        options = helpers._build_voice_options("tts-unknown", {})
        assert len(options) == 1
        assert options[0]["value"] == RECOMMENDED_TTS_VOICE
