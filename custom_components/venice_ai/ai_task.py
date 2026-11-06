"""AI Task integration for Venice AI."""

from __future__ import annotations

import asyncio
import json
import logging

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .ai_task_support import chat_messages_from_log, option_float, option_int
from .client import VeniceAIError
from .const import (
    CONF_CHAT_MODEL,
    CONF_DISABLE_THINKING,
    CONF_MAX_TOKENS,
    CONF_STREAM_RESPONSE,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEVICE_MODEL,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_DISABLE_THINKING,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_STREAM_RESPONSE,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)
from .venice_api import ChatParameters, VeniceConversationService

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
    _LOGGER.debug("Setting up AI Task entities for entry %s", entry.entry_id)
    from . import VeniceAIRuntimeData

    runtime_data: VeniceAIRuntimeData = entry.runtime_data
    if not runtime_data or not runtime_data.client:
        _LOGGER.error(
            "Venice AI client not available in runtime_data for entry %s",
            entry.entry_id,
        )
        return
    entity = VeniceAITaskEntity(entry)
    _LOGGER.debug("Created VeniceAITaskEntity: %s", entity.unique_id)
    # Store entity reference in runtime_data so the service handler can find it
    # without using hass.data (Architecture 7.1 fix)
    runtime_data.ai_task_entity = entity
    async_add_entities([entity])
    _LOGGER.debug("Added VeniceAITaskEntity to Home Assistant")


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
                model=DEVICE_MODEL,
                entry_type=dr.DeviceEntryType.SERVICE,
            )
            # Fix 6: delegate API calls through the service layer so retry
            # logic, metrics, and any future middleware are applied consistently.
            self._service = VeniceConversationService(entry.runtime_data.client)
            self._attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA
            _LOGGER.debug(
                "Initialized VeniceAITaskEntity for entry %s (runtime_data=%s, unique_id=%s)",
                entry.entry_id,
                bool(entry.runtime_data),
                self._attr_unique_id,
            )

        async def async_added_to_hass(self) -> None:
            """Signal that the AI Task entity is fully registered (HIGH-1).

            HA calls this hook once the entity has been added to the state
            machine. Setting the ``ai_task_ready`` event releases any service
            calls that were waiting on the platform setup synchronization
            barrier in ``__init__.py``.
            """
            await super().async_added_to_hass()
            ready = getattr(self.entry.runtime_data, "ai_task_ready", None)
            if ready is not None:
                ready.set()
                _LOGGER.debug(
                    "AI Task entity for entry %s ready; setup barrier released",
                    self.entry.entry_id,
                )


        async def async_generate_data(
            self,
            task: ai_task.GenDataTask,
            chat_log: conversation.ChatLog,
        ) -> ai_task.GenDataTaskResult:
            """Handle a generate data task.

            Public entry-point that service handlers (and HA's ai_task platform)
            should call.  Internally delegates to _async_generate_data so the
            implementation stays testable and overridable.
            """
            return await self._async_generate_data(task, chat_log)

        async def _async_generate_data(
            self,
            task: ai_task.GenDataTask,
            chat_log: conversation.ChatLog,
        ) -> ai_task.GenDataTaskResult:
            """Internal implementation of generate data task."""
            messages = chat_messages_from_log(chat_log.content, task.instructions)

            if not messages or messages[-1].get("role") != "user":
                raise HomeAssistantError("No user message found in chat log")

            options = self.entry.options
            model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            if not isinstance(model, str) or not model.strip():
                model = RECOMMENDED_CHAT_MODEL
            max_tokens = option_int(options, CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
            temperature = option_float(
                options, CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
            )
            top_p = option_float(options, CONF_TOP_P, RECOMMENDED_TOP_P)
            venice_parameters = None
            if options.get(CONF_DISABLE_THINKING, RECOMMENDED_DISABLE_THINKING):
                venice_parameters = {"disable_thinking": True}
            stream = bool(
                options.get(CONF_STREAM_RESPONSE, RECOMMENDED_STREAM_RESPONSE)
            )

            try:
                chat_params = ChatParameters(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    venice_parameters=venice_parameters,
                )
                _LOGGER.debug(
                    "AI task request model=%s stream=%s roles=%s",
                    model,
                    stream,
                    [message.get("role") for message in messages],
                )
                if stream:
                    stream_result = await self._service.chat_stream(
                        messages, chat_params
                    )
                    response_data = stream_result.as_response()
                else:
                    response_data = await self._service.chat(messages, chat_params)

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
            except asyncio.CancelledError:
                raise
            except Exception as err:
                _LOGGER.exception("Unexpected error during task generation")
                raise HomeAssistantError(f"Unexpected error: {err}") from err
