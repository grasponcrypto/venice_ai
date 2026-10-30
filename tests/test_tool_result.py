"""Tool result construction tests.

``ToolResultContent`` changed shape across Home Assistant versions:

* HA 2026.10+: ``ToolResultContent(agent_id, tool_call_id, tool_name,
  result=llm.ToolResult(data, error))`` — ``tool_result`` is no longer an
  init argument, only a deprecated read-only property.
* Earlier HA: ``ToolResultContent(agent_id, tool_name, tool_call_id,
  tool_result)``.
* Oldest HA: ``ToolResultContent(tool_call_id, tool_result)``.

``_make_tool_result_content`` must build a valid object on each of them.
As in ``test_schema.py``, the helpers are AST-extracted from
``conversation.py`` and run against stand-in classes, so Home Assistant
does not need to be installed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


_COMPONENT_DIR = (
    Path(__file__).resolve().parents[1] / "custom_components" / "venice_ai"
)


@dataclass(slots=True)
class _ToolResult:
    """Stand-in for ``homeassistant.helpers.llm.ToolResult`` (HA 2026.10+)."""

    data: Any
    error: bool = False


@dataclass
class _ContentHA202610:
    agent_id: str
    tool_call_id: str
    tool_name: str
    result: _ToolResult


@dataclass
class _ContentPre202610:
    agent_id: str
    tool_name: str
    tool_call_id: str
    tool_result: Any


@dataclass
class _ContentOldest:
    tool_call_id: str
    tool_result: Any
    extra: dict = field(default_factory=dict)


def _load(content_cls: type, llm: Any) -> dict[str, Callable[..., Any]]:
    """Exec the tool-result helpers with the given HA stand-ins."""
    target_names = {"_tool_result_data", "_make_tool_result_content"}
    source = (_COMPONENT_DIR / "conversation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in target_names
    ]
    assert {fn.name for fn in selected} == target_names
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "Any": Any,
        "DOMAIN": "venice_ai",
        "ToolResultContent": content_cls,
        "llm": llm,
    }
    code = compile(ast.Module(body=selected, type_ignores=[]), "conversation.py", "exec")
    exec(code, namespace)
    return namespace


_LLM_NEW = SimpleNamespace(ToolResult=_ToolResult)
_LLM_OLD = SimpleNamespace()


class TestHA202610:
    """HA 2026.10+: ``result=ToolResult(...)``."""

    def test_dict_result_is_wrapped(self) -> None:
        make = _load(_ContentHA202610, _LLM_NEW)["_make_tool_result_content"]
        content = make("HassTurnOff", "call_1", {"speech": "done"})
        assert content == _ContentHA202610(
            agent_id="venice_ai",
            tool_call_id="call_1",
            tool_name="HassTurnOff",
            result=_ToolResult(data={"speech": "done"}, error=False),
        )

    def test_error_dict_sets_error_flag(self) -> None:
        make = _load(_ContentHA202610, _LLM_NEW)["_make_tool_result_content"]
        content = make("HassTurnOff", "call_1", {"error": "No entities matched"})
        assert content.result.error is True

    def test_tool_result_passes_through(self) -> None:
        make = _load(_ContentHA202610, _LLM_NEW)["_make_tool_result_content"]
        result = _ToolResult(data={"result": "Live Context: ..."})
        content = make("GetLiveContext", "call_2", result)
        assert content.result is result


class TestOlderHA:
    """Pre-2026.10 signatures still get the plain JSON payload."""

    def test_four_arg_signature(self) -> None:
        make = _load(_ContentPre202610, _LLM_OLD)["_make_tool_result_content"]
        content = make("HassTurnOn", "call_1", {"speech": "ok"})
        assert content == _ContentPre202610(
            agent_id="venice_ai",
            tool_name="HassTurnOn",
            tool_call_id="call_1",
            tool_result={"speech": "ok"},
        )

    def test_two_arg_signature(self) -> None:
        make = _load(_ContentOldest, _LLM_OLD)["_make_tool_result_content"]
        content = make("HassTurnOn", "call_1", {"speech": "ok"})
        assert content.tool_call_id == "call_1"
        assert content.tool_result == {"speech": "ok"}

    def test_tool_result_object_is_unwrapped(self) -> None:
        # Intent tools may already return a ToolResult while the content
        # class still expects the raw payload.
        make = _load(_ContentPre202610, _LLM_NEW)["_make_tool_result_content"]
        content = make("GetLiveContext", "call_2", _ToolResult(data={"a": 1}))
        assert content.tool_result == {"a": 1}


@pytest.mark.parametrize(
    ("value", "expected"),
    [({"a": 1}, {"a": 1}), (_ToolResult(data={"a": 1}), {"a": 1})],
)
def test_tool_result_data(value: Any, expected: Any) -> None:
    data = _load(_ContentHA202610, _LLM_NEW)["_tool_result_data"]
    assert data(value) == expected
