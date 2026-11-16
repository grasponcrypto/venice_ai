"""Regression tests for malformed LLM tool-call protocol recovery."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable

import pytest


_CONVERSATION = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "venice_ai"
    / "conversation.py"
)


def _load_normalizer() -> Callable[..., Any]:
    """Load the pure tool-call normalizer without importing Home Assistant."""
    tree = ast.parse(_CONVERSATION.read_text(encoding="utf-8"))
    selected = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_tool_call"
    )
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "Any": Any,
        "json": json,
        "_INVALID_TOOL_NAME": "__invalid_tool_call__",
    }
    exec(
        compile(ast.Module(body=[selected], type_ignores=[]), "conversation.py", "exec"),
        namespace,
    )
    return namespace["_normalize_tool_call"]


@pytest.fixture
def normalize_tool_call() -> Callable[..., Any]:
    return _load_normalizer()


@pytest.mark.parametrize(
    ("raw_call", "expected_error"),
    [
        (
            {"type": "function", "function": {"name": "turn_on", "arguments": "{}"}},
            "missing call id",
        ),
        (
            {
                "id": "call_1",
                "type": "not_function",
                "function": {"name": "turn_on", "arguments": "{}"},
            },
            "invalid call type",
        ),
        ({"id": "call_1"}, "missing function data"),
        (
            {"id": "call_1", "function": {"arguments": "{}"}},
            "missing function name",
        ),
        (
            {"id": "call_1", "function": {"name": "turn_on", "arguments": "{"}},
            "arguments are not valid JSON",
        ),
    ],
)
def test_malformed_calls_become_pairable(
    normalize_tool_call: Callable[..., Any],
    raw_call: dict[str, Any],
    expected_error: str,
) -> None:
    normalized, error = normalize_tool_call(raw_call, 2, 3)

    assert error is not None
    assert expected_error in error
    assert normalized["id"]
    assert normalized["type"] == "function"
    assert isinstance(normalized["function"], dict)
    assert normalized["function"]["name"]
    assert isinstance(normalized["function"]["arguments"], str)


def test_missing_id_is_deterministic_for_assistant_tool_pair(
    normalize_tool_call: Callable[..., Any],
) -> None:
    raw_calls = [
        {"function": {"name": "turn_on", "arguments": "{}"}},
        {"id": "call_2", "function": {"name": "turn_off", "arguments": "{}"}},
    ]

    normalized_calls = [
        normalize_tool_call(raw_call, 0, index)[0]
        for index, raw_call in enumerate(raw_calls)
    ]
    assistant_ids = [call["id"] for call in normalized_calls]
    tool_result_ids = [call["id"] for call in normalized_calls]

    assert assistant_ids == tool_result_ids
    assert assistant_ids[0] == "venice_malformed_tool_call_0_0"


def test_valid_call_is_unchanged_and_has_no_error(
    normalize_tool_call: Callable[..., Any],
) -> None:
    raw_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "turn_on", "arguments": '{"entity_id":"light.kitchen"}'},
    }

    normalized, error = normalize_tool_call(raw_call, 0, 0)

    assert normalized == raw_call
    assert error is None


def test_only_valid_call_reaches_executor(
    normalize_tool_call: Callable[..., Any],
) -> None:
    raw_calls = [
        {"id": "bad", "function": {"name": "turn_on", "arguments": "{"}},
        {"id": "good", "function": {"name": "turn_on", "arguments": "{}"}},
    ]

    prepared = [
        normalize_tool_call(raw_call, 0, index)
        for index, raw_call in enumerate(raw_calls)
    ]
    executor_calls = [
        call["function"]["name"]
        for call, error in prepared
        if error is None
    ]

    assert executor_calls == ["turn_on"]
