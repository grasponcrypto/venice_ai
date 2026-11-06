"""AI Task helpers must match Home Assistant 2026.10 and older cores."""

from __future__ import annotations

from dataclasses import dataclass

from .conftest import load_component_module

support = load_component_module("ai_task_support")


@dataclass
class _Core202610Task:
    name: str
    instructions: str
    structure: object | None = None


@dataclass
class _OlderTask:
    instructions: str
    structure: object | None = None


def test_2026_10_task_requires_name_and_drops_empty_structure() -> None:
    task = support.build_gen_data_task(
        _Core202610Task,
        "return the word ok",
        "",
    )
    assert task.name == "return the word ok"
    assert task.instructions == "return the word ok"
    assert task.structure is None


def test_older_task_does_not_require_name() -> None:
    task = support.build_gen_data_task(_OlderTask, "return the word ok", None)
    assert task.instructions == "return the word ok"
    assert "name" not in task.__dataclass_fields__
    assert task.structure is None


def test_nonempty_structure_is_kept() -> None:
    task = support.build_gen_data_task(
        _OlderTask,
        "return json",
        '{"ok": "boolean"}',
    )
    assert task.structure == '{"ok": "boolean"}'


def test_blank_instructions_still_get_a_name() -> None:
    task = support.build_gen_data_task(_Core202610Task, "   ", None)
    assert task.name == "Venice AI task"


def test_append_skips_instruction_already_in_the_chat_log() -> None:
    messages = [{"role": "user", "content": "return the word ok"}]
    support.append_task_instruction(messages, "return the word ok")
    assert messages == [{"role": "user", "content": "return the word ok"}]


def test_append_adds_instruction_when_missing() -> None:
    messages = [{"role": "system", "content": "You are a Home Assistant expert."}]
    support.append_task_instruction(messages, "return the word ok")
    assert messages[-1] == {"role": "user", "content": "return the word ok"}


class _LogMessage:
    def __init__(self, role: str, content: object) -> None:
        self.role = role
        self.content = content


def test_chat_messages_drop_empty_system_and_keep_the_task() -> None:
    messages = support.chat_messages_from_log(
        [
            _LogMessage("system", ""),
            _LogMessage("system", "You are a Home Assistant expert."),
            _LogMessage("user", "return the word OK."),
        ],
        "return the word OK.",
    )
    assert messages == [
        {"role": "system", "content": "You are a Home Assistant expert."},
        {"role": "user", "content": "return the word OK."},
    ]


def test_option_numbers_reject_junk_and_truncate_floats() -> None:
    options = {"max_tokens": 512.0, "temperature": "hot", "top_p": True}
    assert support.option_int(options, "max_tokens", 128) == 512
    assert support.option_float(options, "temperature", 1.0) == 1.0
    assert support.option_float(options, "top_p", 0.9) == 0.9
