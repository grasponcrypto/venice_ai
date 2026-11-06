"""Helpers for the AI Task service that do not import Home Assistant.

Home Assistant 2026.10 made ``GenDataTask.name`` required and stopped
letting integrations build the chat log themselves. These helpers keep the
service working on that core and on older cores that still take only
``instructions``.
"""

from __future__ import annotations

import inspect
from typing import Any


def task_name(instructions: str) -> str:
    """Return a short task name from the first line of the instructions."""
    stripped = instructions.strip()
    if not stripped:
        return "Venice AI task"
    return stripped.splitlines()[0].strip()[:80] or "Venice AI task"


def build_gen_data_task(
    task_cls: type,
    instructions: str,
    structure: Any = None,
    *,
    name: str | None = None,
) -> Any:
    """Build a ``GenDataTask`` for whichever core is installed.

    An empty structure string is omitted. Passing ``""`` used to be harmless,
    but a required ``name`` field made ``GenDataTask(instructions=...)`` raise
    ``TypeError`` before the service handler could record the failure.
    """
    params = inspect.signature(task_cls).parameters
    kwargs: dict[str, Any] = {}
    if "name" in params:
        kwargs["name"] = name or task_name(instructions)
    if "instructions" in params:
        kwargs["instructions"] = instructions
    elif "task" in params:
        kwargs["task"] = instructions
    else:
        raise TypeError(f"{task_cls.__name__} has no instructions field")

    if "structure" in params and _structure_provided(structure):
        kwargs["structure"] = structure
    return task_cls(**kwargs)


def chat_messages_from_log(content: object, instructions: str) -> list[dict[str, str]]:
    """Convert a chat log into Venice messages.

    An empty system prompt is dropped. Venice rejects it as invalid parameters.
    The task text is added only when the log does not already end with it.
    """
    messages: list[dict[str, str]] = []
    items = content if isinstance(content, list) else []
    for msg in items:
        role = getattr(msg, "role", None)
        raw = getattr(msg, "content", None)
        if isinstance(raw, str):
            text = raw
        elif raw is None:
            text = ""
        else:
            text = str(raw)
        if role == "system":
            if text.strip():
                messages.append({"role": "system", "content": text})
        elif role == "user":
            if text.strip():
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": text})
    append_task_instruction(messages, instructions)
    return messages


def option_int(options: object, key: str, default: int) -> int:
    """Return an integer option. Floats are truncated; junk uses the default."""
    if not isinstance(options, dict):
        return default
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def option_float(options: object, key: str, default: float) -> float:
    """Return a float option. Junk uses the default."""
    if not isinstance(options, dict):
        return default
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def append_task_instruction(messages: list[dict[str, Any]], instructions: str) -> None:
    """Add the task text unless the chat log already ends with it.

    ``internal_async_generate_data`` already inserts the instructions as the
    last user message. Appending them again sends the prompt twice.
    """
    if (
        messages
        and messages[-1].get("role") == "user"
        and messages[-1].get("content") == instructions
    ):
        return
    messages.append({"role": "user", "content": instructions})


def _structure_provided(structure: Any) -> bool:
    """Return True when the caller actually supplied a structure."""
    if structure is None:
        return False
    if isinstance(structure, str) and not structure.strip():
        return False
    return True
