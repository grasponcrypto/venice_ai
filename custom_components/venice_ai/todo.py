"""Venice AI Task Entity for generating AI-powered tasks."""

from __future__ import annotations

import json
from typing import Any

from homeassistant.components.todo import AITaskEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .client import AsyncVeniceAIClient, VeniceAIError
from .const import LOGGER
from .task_types import VeniceTask


class VeniceAITaskEntity(AITaskEntity):
    """Venice AI Task Entity for generating tasks using AI."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the Venice AI Task Entity."""
        super().__init__(entry)
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_task"
        self._client: AsyncVeniceAIClient = entry.runtime_data

    async def _async_generate_data(
        self, prompt: str | None = None
    ) -> list[dict[str, Any]]:
        """Generate tasks using Venice AI based on the provided prompt."""
        if not prompt:
            prompt = (
                "Generate a list of 5 useful daily tasks for smart home user."
            )

        try:
            # Prepare the message for Venice AI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Generate task lists as JSON arrays with 'summary', "
                        "optional 'description' and 'due_date'."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            # Call Venice AI API
            response_data = await self._client.chat.create_non_streaming({
                "model": "default",  # Use default model
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": False,
            })

            if not response_data or not response_data.get("choices"):
                LOGGER.error("Invalid Venice AI response: %s", response_data)
                raise HomeAssistantError("Invalid Venice AI response")

            content = (
                response_data["choices"][0]
                .get("message", {})
                .get("content", "")
            )
            if not content:
                LOGGER.error("No content in Venice AI response")
                raise HomeAssistantError("No content received from Venice AI")

            # Parse the JSON response
            try:
                tasks_data = json.loads(content)
                if not isinstance(tasks_data, list):
                    raise ValueError("Response is not a list")
            except (json.JSONDecodeError, ValueError) as err:
                LOGGER.error("Failed to parse Venice AI tasks: %s", content)
                raise HomeAssistantError(f"Parse error: {err}") from err

            # Convert to VeniceTask objects and then to dicts
            tasks = []
            for i, task_data in enumerate(tasks_data):
                if (
                    not isinstance(task_data, dict)
                    or "summary" not in task_data
                ):
                    continue
                task = VeniceTask(
                    uid=f"venice_task_{i}",
                    summary=task_data["summary"],
                    description=task_data.get("description"),
                    due_date=task_data.get("due_date"),
                )
                tasks.append(task.to_dict())

            return tasks

        except VeniceAIError as err:
            LOGGER.error("Venice AI error during task generation: %s", err)
            raise HomeAssistantError(f"Error generating tasks: {err}") from err
        except Exception as err:
            LOGGER.exception("Unexpected error during task generation")
            raise HomeAssistantError(f"Unexpected error: {err}") from err


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities,
) -> None:
    """Set up Venice AI Task Entity from a config entry."""
    if not entry.runtime_data:
        LOGGER.error(
            "Venice AI client not available in runtime_data for entry %s",
            entry.entry_id
        )
        return
    entity = VeniceAITaskEntity(entry)
    async_add_entities([entity])