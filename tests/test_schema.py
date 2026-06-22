"""Schema conversion tests (TEST-3).

These tests exercise the schema-conversion helpers in ``conversation.py``:

* ``_format_venice_schema`` — produces Venice-compatible OpenAPI schemas.
* ``_convert_schema_to_hashable`` — converts voluptuous schemas into a
  hashable representation suitable for ``voluptuous_openapi.convert``.

Rather than importing ``conversation.py`` directly (which transitively
imports Home Assistant and is not available in a plain unit-test
environment), the helpers are extracted from the source file via AST and
executed in a clean namespace. This keeps the tests faithful to the actual
implementation while still running under the lightweight pytest harness
that ``tests/conftest.py`` sets up.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Callable, cast

import pytest


_COMPONENT_DIR = (
    Path(__file__).resolve().parents[1] / "custom_components" / "venice_ai"
)


class _StubSelector:
    """Placeholder for ``homeassistant.helpers.selector.Selector`` subclasses."""

    SelectSelector = type("SelectSelector", (), {})


class _StubSelectorModule:
    """Drop-in module stand-in for ``homeassistant.helpers.selector``."""

    Selector = _StubSelector
    SelectSelector = _StubSelector.SelectSelector
    BooleanSelector = type("BooleanSelector", (), {})
    NumberSelector = type("NumberSelector", (), {})
    TemplateSelector = type("TemplateSelector", (), {})


def _extract_helpers() -> dict[str, object]:
    """AST-extract the schema helpers from ``conversation.py`` for isolated testing.

    Only module-level function definitions whose name is one of the helpers
    we want to test are loaded. The body of each function is exec'd in a
    fresh namespace that does not have access to ``homeassistant`` or any
    other HA imports, which is fine because these helpers are pure
    transformations that don't touch HA state. Names that the helpers
    reference at module level (``Any``, ``selector``, ``_LOGGER``) are
    injected into the namespace so the bodies can run unchanged.
    """
    target_names = {"_format_venice_schema", "_convert_schema_to_hashable"}
    source = (_COMPONENT_DIR / "conversation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected: list[ast.FunctionDef] = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in target_names
    ]
    if {fn.name for fn in selected} != target_names:
        missing = target_names - {fn.name for fn in selected}
        raise AssertionError(f"Could not find helpers in conversation.py: {missing}")

    namespace: dict[str, object] = {
        "__builtins__": __builtins__,
        # Names referenced inside the helper bodies (from module-level imports).
        "Any": Any,
        "selector": _StubSelectorModule,
        "_LOGGER": logging.getLogger("venice_ai.test_schema"),
    }
    for fn in selected:
        snippet = ast.Module(body=[fn], type_ignores=[])
        code = compile(snippet, filename=f"conversation.py:{fn.name}", mode="exec")
        exec(code, namespace)
    return namespace


HELPERS: dict[str, object] = _extract_helpers()
_format_venice_schema = cast(Callable[[Any], Any], HELPERS["_format_venice_schema"])
_convert_schema_to_hashable = cast(
    Callable[[Any], Any], HELPERS["_convert_schema_to_hashable"]
)


class TestFormatVeniceSchema:
    """Validate the per-key type mapping in ``_format_venice_schema``."""

    def test_string_mapping(self) -> None:
        schema = _format_venice_schema({"name": str})
        assert schema == {"name": {"type": "string"}}

    def test_int_mapping(self) -> None:
        schema = _format_venice_schema({"count": int})
        assert schema == {"count": {"type": "integer"}}

    def test_float_mapping(self) -> None:
        schema = _format_venice_schema({"score": float})
        assert schema == {"score": {"type": "number"}}

    def test_bool_mapping(self) -> None:
        schema = _format_venice_schema({"enabled": bool})
        assert schema == {"enabled": {"type": "boolean"}}

    def test_unknown_type_defaults_to_string(self) -> None:
        class Custom:
            pass

        schema = _format_venice_schema({"thing": Custom})
        assert schema == {"thing": {"type": "string"}}

    def test_multiple_keys(self) -> None:
        schema = _format_venice_schema({"a": str, "b": int})
        assert schema == {"a": {"type": "string"}, "b": {"type": "integer"}}

    def test_empty_schema(self) -> None:
        assert _format_venice_schema({}) == {}


class TestConvertSchemaToHashable:
    """Validate the hashable conversion used by ``voluptuous_openapi``."""

    def test_dict_becomes_frozenset_of_items(self) -> None:
        result = _convert_schema_to_hashable({"a": str})
        assert isinstance(result, frozenset)
        members = list(result)
        assert ("a", str) in members

    def test_list_becomes_tuple(self) -> None:
        result = _convert_schema_to_hashable([str, int])
        assert isinstance(result, tuple)
        assert result == (str, int)

    def test_plain_type_passthrough(self) -> None:
        assert _convert_schema_to_hashable(str) is str
        assert _convert_schema_to_hashable(int) is int

    def test_nested_dict_and_list(self) -> None:
        result = _convert_schema_to_hashable({"items": [str]})
        assert isinstance(result, frozenset)
        members = dict(result)
        assert isinstance(members["items"], tuple)
        assert members["items"] == (str,)

    def test_empty_dict(self) -> None:
        result = _convert_schema_to_hashable({})
        assert isinstance(result, frozenset)
        assert len(result) == 0


@pytest.mark.parametrize(
    ("input_value", "expected_type"),
    [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
    ],
)
def test_schema_type_mapping_param(input_value: type, expected_type: str) -> None:
    """Parametrised regression test for the JSON-schema type mapping."""
    schema = _format_venice_schema({"value": input_value})
    assert schema["value"]["type"] == expected_type
