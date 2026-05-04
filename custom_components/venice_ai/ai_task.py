"""AI Task integration for Venice AI."""

from __future__ import annotations

import json
import logging

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import AsyncVeniceAIClient, VeniceAIError
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
)

_LOGGER = logging.getLogger(__name__)

try:
    from homeassistant.components import ai_task
    _HAS_AI_TASK = True
except ImportError:
    ai_task = None  # type: ignore[assignment]
    _HAS_AI_TASK = False


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    if not _HAS_AI_TASK:
        _LOGGER.warning(
            "AI Task platform is not available in this Home Assistant version"
        )
        return
    _LOGGER.info("Setting up AI Task entities for entry %s", entry.entry_id)
    from . import VeniceAIRuntimeData

    runtime_data: VeniceAIRuntimeData = entry.runtime_data
    if not runtime_data or not runtime_data.client:
        _LOGGER.error(
            "Venice AI client not available in runtime_data for entry %s",
            entry.entry_id,
        )
        return
    entity = VeniceAITaskEntity(entry)
    _LOGGER.info("Created VeniceAITaskEntity: %s", entity.unique_id)
    async_add_entities([entity])
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entity
    _LOGGER.info("Added VeniceAITaskEntity to Home Assistant")


if not _HAS_AI_TASK:
    # Define a dummy class so the module is import-safe when ai_task is unavailable
    class _DummyAITaskEntity:
        """Placeholder when ai_task is unavailable."""

    VeniceAITaskEntity = _DummyAITaskEntity  # type: ignore[misc,assignment]
else:

    class VeniceAITaskEntity(ai_task.AITaskEntity):
        """Venice AI AI Task entity."""

        _attr_has_entity_name = True
        _attr_name = "AI Task"

        def __init__(self, entry: ConfigEntry) -> None:
            """Initialize the entity."""
            super().__init__()
            self.entry = entry
            self._attr_unique_id = f"{entry.entry_id}_task"
            self._attr_device_info = dr.DeviceInfo(
                identifiers={(DOMAIN, entry.entry_id)},
                name=entry.title,
                manufacturer="Venice AI",
                model="AI Task",
                entry_type=dr.DeviceEntryType.SERVICE,
            )
            self._client: AsyncVeniceAIClient = entry.runtime_data.client
            self._attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA
            _LOGGER.info(
                "Initialized VeniceAITaskEntity for entry %s (runtime_data=%s, unique_id=%s)",
                entry.entry_id,
                bool(entry.runtime_data),
                self._attr_unique_id,
            )

        async def _async_generate_data(
            self,
            task: ai_task.GenDataTask,
            chat_log: conversation.ChatLog,
        ) -> ai_task.GenDataTaskResult:
            """Handle a generate data task."""
            # Build a local messages list without mutating chat_log.content
            messages = []
            for msg in chat_log.content:
                if isinstance(msg, conversation.SystemContent):
                    messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, conversation.UserContent):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, conversation.AssistantContent):
                    venice_msg = {
                        "role": "assistant",
                        "content": msg.content or "",
                    }
                    messages.append(venice_msg)

            # Append task instructions as a user message in the local list
            messages.append({"role": "user", "content": task.instructions})

            if not messages or messages[-1].get("role") != "user":
                raise HomeAssistantError("No user message found in chat log")

            # Use the configured chat model from options
            model = self.entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

            # Use configured options from config entry instead of hardcoded values
            max_tokens = self.entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
            temperature = self.entry.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)

            try:
                response_data = await self._client.chat.completions.create_non_streaming(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )

                if not response_data or not response_data.get("choices"):
                    raise HomeAssistantError("Invalid Venice AI response")

                text = (
                    response_data["choices"][0]
                    .get("message", {})
                    .get("content", "")
                )

                if not task.structure:
                    return ai_task.GenDataTaskResult(
                        conversation_id=chat_log.conversation_id,
                        data=text,
                    )

                try:
                    data = json.loads(text)
                except json.JSONDecodeError as err:
                    _LOGGER.error(
                        "Failed to parse JSON response: %s. Response: %s",
                        err,
                        text,
                    )
                    raise HomeAssistantError(
                        "Error parsing structured response"
                    ) from err

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
