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
