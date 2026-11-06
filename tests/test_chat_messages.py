"""Chat-log conversion: one system prompt, one user turn, replayable tool calls."""

from __future__ import annotations

import ast
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

_COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "venice_ai"
_HELPERS = {
    "_strip_thinking",
    "_tool_result_data",
    "_decode_legacy_tool_payload",
    "_venice_tool_calls_from_inputs",
    "_tool_inputs_from_venice_calls",
    "_make_assistant_content",
    "_user_message_already_present",
    "_convert_chat_log_to_venice_messages",
    "_is_probatio_schema",
    "_ensure_schema_types",
    "_normalize_tool_schema",
    "_schema_from_voluptuous",
    "_schema_from_probatio",
    "_format_venice_schema",
    "_convert_tool_parameters",
}


class _Content:
    def __init__(self, content=None, **kwargs):
        self.content = content
        for key, value in kwargs.items():
            setattr(self, key, value)


class UserContent(_Content):
    pass


class AssistantContent(_Content):
    def __init__(self, agent_id=None, content=None, tool_calls=None):
        super().__init__(content=content, agent_id=agent_id, tool_calls=tool_calls)


class SystemContent(_Content):
    pass


class ToolResultContent(_Content):
    def __init__(self, tool_call_id=None, tool_result=None, **kwargs):
        super().__init__(**kwargs)
        self.tool_call_id = tool_call_id
        self.tool_result = tool_result


class _ToolInput:
    def __init__(self, id="", tool_name="", tool_args=None):
        self.id = id
        self.tool_name = tool_name
        self.tool_args = tool_args or {}


class _ChatLog:
    def __init__(self, content):
        self.content = content


def _load():
    source = (_COMPONENT / "conversation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _HELPERS
    ]
    missing = _HELPERS - {node.name for node in selected}
    if missing:
        raise AssertionError(f"missing helpers: {missing}")
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, *selected], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "Any": Any,
        "json": json,
        "_LOGGER": logging.getLogger("venice_ai.test_chat"),
        "UserContent": UserContent,
        "AssistantContent": AssistantContent,
        "SystemContent": SystemContent,
        "ToolResultContent": ToolResultContent,
        "DOMAIN": "venice_ai",
        "llm": types.SimpleNamespace(ToolInput=_ToolInput),
        "voluptuous_convert": lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected voluptuous convert")
        ),
    }
    exec(compile(module, "conversation.py", "exec"), namespace)
    return namespace


NS = _load()
convert = NS["_convert_chat_log_to_venice_messages"]
already = NS["_user_message_already_present"]
make_assistant = NS["_make_assistant_content"]
convert_tool = NS["_convert_tool_parameters"]


def test_system_prompt_is_not_sent_twice() -> None:
    prompt = "You are a helpful smart home assistant."
    log = _ChatLog([SystemContent(prompt), UserContent("turn on the light")])
    messages = convert(log, prompt)
    assert messages == [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "turn on the light"},
    ]


def test_user_turn_already_logged_by_home_assistant_is_not_appended() -> None:
    log = _ChatLog([SystemContent("prompt"), UserContent("hello")])
    assert already(log, "hello") is True
    assert already(log, "a new sentence") is False


def test_new_user_turn_after_an_assistant_reply_is_appended() -> None:
    log = _ChatLog([UserContent("hello"), AssistantContent(content="hi")])
    assert already(log, "hello") is False


def test_structured_tool_calls_keep_the_call_id() -> None:
    tool_input = _ToolInput(
        id="call_abc",
        tool_name="HassTurnOn",
        tool_args={"name": "Kitchen"},
    )
    log = _ChatLog(
        [
            SystemContent("prompt"),
            UserContent("turn on the kitchen light"),
            AssistantContent(content="", tool_calls=[tool_input]),
            ToolResultContent(tool_call_id="call_abc", tool_result={"success": True}),
        ]
    )
    messages = convert(log, "prompt")
    assistant = messages[2]
    assert assistant["tool_calls"][0]["id"] == "call_abc"
    assert assistant["tool_calls"][0]["function"]["name"] == "HassTurnOn"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Kitchen"
    }
    assert messages[3]["tool_call_id"] == "call_abc"
    assert assistant["content"] == ""


def test_legacy_json_payload_still_replays_and_plain_json_stays_text() -> None:
    payload = json.dumps(
        {
            "text": "checking",
            "tool_calls": [
                {
                    "id": "call_legacy",
                    "type": "function",
                    "function": {"name": "HassTurnOn", "arguments": "{}"},
                }
            ],
        }
    )
    log = _ChatLog([AssistantContent(content=payload)])
    messages = convert(log, "")
    assert messages[0]["content"] == "checking"
    assert messages[0]["tool_calls"][0]["id"] == "call_legacy"

    plain = _ChatLog([AssistantContent(content='{"answer": 1}')])
    assert convert(plain, "") == [{"role": "assistant", "content": '{"answer": 1}'}]



def test_thinking_tags_are_stripped_after_the_tool_payload_is_decoded() -> None:
    payload = json.dumps(
        {
            "text": "<think>secret</think>done",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "HassTurnOn", "arguments": "{}"},
                }
            ],
        }
    )
    log = _ChatLog([AssistantContent(content=payload)])
    messages = convert(log, "", strip_thinking=True)
    assert messages[0]["content"] == "done"
    assert messages[0]["tool_calls"][0]["id"] == "call_1"


def test_make_assistant_content_uses_tool_calls_field() -> None:
    stored = make_assistant(
        "checking",
        [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "HassTurnOn", "arguments": '{"name": "Kitchen"}'},
            }
        ],
    )
    assert stored.tool_calls[0].id == "call_abc"
    assert stored.content == "checking"
    assert not str(stored.content).startswith("{")


def test_probatio_schema_is_converted_with_the_serializer() -> None:
    seen = {}

    class Schema:
        pass

    Schema.__module__ = "probatio"

    schema = Schema()

    def to_openapi(parameters, custom_serializer=None, openapi_version=None):
        seen["parameters"] = parameters
        seen["serializer"] = custom_serializer
        seen["version"] = openapi_version
        return {
            "type": "object",
            "properties": {"name": {"type": "string", "format": "entity_id"}},
            "required": ["name"],
        }

    fake = types.ModuleType("probatio")
    fake.to_openapi = to_openapi
    sys.modules["probatio"] = fake
    try:
        tool = types.SimpleNamespace(name="HassTurnOn", parameters=schema)
        converted = convert_tool(tool, custom_serializer="serializer")
    finally:
        sys.modules.pop("probatio", None)

    assert seen["parameters"] is schema
    assert seen["serializer"] == "serializer"
    assert converted["properties"]["name"]["format"] == "entity_id"


def test_voluptuous_schema_is_not_rewritten_before_conversion() -> None:
    original = object()
    seen = {}

    def fake_convert(parameters, custom_serializer=None):
        seen["parameters"] = parameters
        seen["serializer"] = custom_serializer
        return {"type": "object", "properties": {"q": {"type": "string"}}}

    NS["voluptuous_convert"] = fake_convert
    tool = types.SimpleNamespace(name="search", parameters=original)
    converted = convert_tool(tool, custom_serializer="selector")
    assert seen["parameters"] is original
    assert seen["serializer"] == "selector"
    assert converted["properties"]["q"]["type"] == "string"

