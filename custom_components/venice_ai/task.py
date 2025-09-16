"""AI Task integration for Venice AI."""

from __future__ import annotations

import json
import logging
from typing import Any

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import AsyncVeniceAIClient, VeniceAIError
from .const import LOGGER

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    if not entry.runtime_data:
        LOGGER.error(
            "Venice AI client not available in runtime_data for entry %s",
            entry.entry_id
        )
        return
    async_add_entities([VeniceAITaskEntity(entry)])


class VeniceAITaskEntity(ai_task.AITaskEntity):
    """Venice AI AI Task entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        super().__init__(entry)
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_task"
        self._client: AsyncVeniceAIClient = entry.runtime_data
        self._attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        # Provide LLM data to chat_log
        await chat_log.async_provide_llm_data(
            task.as_llm_context(),
            [],  # No LLM API IDs for now
            None,  # No system prompt override
            None,  # No extra system prompt
        )

        # Convert chat_log content to Venice messages
        messages = []
        for msg in chat_log.content:
            if isinstance(msg, conversation.SystemContent):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, conversation.UserContent):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, conversation.AssistantContent):
                venice_msg = {"role": "assistant", "content": msg.content or ""}
                messages.append(venice_msg)
            # Skip other types for now

        if not messages or messages[-1].get("role") != "user":
            raise HomeAssistantError("No user message found in chat log")

        # Call Venice AI
        try:
            response_data = await self._client.chat.create_non_streaming({
                "model": "default",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": False,
            })

            if not response_data or not response_data.get("choices"):
                raise HomeAssistantError("Invalid Venice AI response")

            text = response_data["choices"][0].get("message", {}).get("content", "")

            if not task.structure:
                return ai_task.GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=text,
                )

            # Parse structured data
            try:
                data = json.loads(text)
            except json.JSONDecodeError as err:
                _LOGGER.error("Failed to parse JSON response: %s. Response: %s", err, text)
                raise HomeAssistantError("Error parsing structured response") from err

            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=data,
            )

        except VeniceAIError as err:
            _LOGGER.error("Venice AI error during task generation: %s", err)
            raise HomeAssistantError(f"Error generating data: {err}") from err
        except Exception as err:
            _LOGGER.exception("Unexpected error during task generation")
            raise HomeAssistantError(f"Unexpected error: {err}") from err