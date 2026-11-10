"""Tests for Home Assistant LLM API discovery fallback behavior."""

from __future__ import annotations

import ast
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIG_FLOW_PATH = ROOT / "custom_components" / "venice_ai" / "config_flow.py"


def _load_fallback_helper() -> types.ModuleType:
    """Load the pure fallback helper without importing Home Assistant."""
    tree = ast.parse(CONFIG_FLOW_PATH.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_assist_fallback"
    )

    module = types.ModuleType("config_flow_llm_api_helpers")

    def select_option_dict(*, label: str, value: str) -> dict[str, str]:
        return {"label": label, "value": value}

    module.__dict__["SelectOptionDict"] = select_option_dict
    module.__dict__["_LOGGER"] = types.SimpleNamespace(debug=lambda *args: None)
    module.__dict__["list"] = list
    exec(  # noqa: S102 - execute the AST-isolated helper under test.
        compile(ast.Module(body=[helper], type_ignores=[]), str(CONFIG_FLOW_PATH), "exec"),
        module.__dict__,
    )
    return module


def test_fetch_llm_api_options_always_applies_fallback() -> None:
    tree = ast.parse(CONFIG_FLOW_PATH.read_text(encoding="utf-8"))
    fetch_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_fetch_llm_api_options"
    )

    fallback_calls = [
        node
        for node in ast.walk(fetch_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_ensure_assist_fallback"
    ]

    assert len(fallback_calls) == 1
    assert isinstance(fetch_method.body[-2], ast.Expr)
    final_call = fetch_method.body[-2].value
    assert isinstance(final_call, ast.Call)
    assert isinstance(final_call.func, ast.Name)
    assert final_call.func.id == "_ensure_assist_fallback"


def test_empty_discovery_adds_assist_fallback() -> None:
    helper = _load_fallback_helper()

    options: list[dict[str, Any]] = []
    result = helper._ensure_assist_fallback(options)

    assert result == [{"label": "Assist", "value": "assist"}]


def test_single_discovered_api_is_not_duplicated() -> None:
    helper = _load_fallback_helper()
    options = [{"label": "OpenAI", "value": "openai"}]

    result = helper._ensure_assist_fallback(options)

    assert result == options


def test_existing_assist_api_is_not_duplicated() -> None:
    helper = _load_fallback_helper()
    options = [{"label": "Assist", "value": "assist"}]

    result = helper._ensure_assist_fallback(options)

    assert result == options
