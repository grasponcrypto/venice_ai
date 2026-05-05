"""The Venice AI Conversation integration."""

from __future__ import annotations

import logging

import voluptuous as vol

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.components import conversation

# Conditional import for ai_task (availability depends on HA version)
try:
    from homeassistant.components import ai_task
    _HAS_AI_TASK = True
except ImportError:
    _HAS_AI_TASK = False

from .client import AsyncVeniceAIClient, VeniceAIError, AuthenticationError
from .const import DOMAIN, HAS_VOLUPTUOUS_OPENAPI
from .coordinator import VeniceAIDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_AI_TASK = "ai_task"
PLATFORMS = [Platform.CONVERSATION, Platform.TTS, Platform.STT]
if _HAS_AI_TASK:
    ai_task_platform = getattr(Platform, "AI_TASK", None)
    if ai_task_platform:
        PLATFORMS.append(ai_task_platform)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass
class VeniceAIRuntimeData:
    """Runtime data stored in the config entry."""

    client: AsyncVeniceAIClient
    coordinator: VeniceAIDataUpdateCoordinator
    ai_task_entity: object | None = None


class VeniceAIConfigEntry(ConfigEntry):
    """Venice AI config entry with runtime data."""

    runtime_data: VeniceAIRuntimeData


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Venice AI Conversation."""
    if not HAS_VOLUPTUOUS_OPENAPI:
        _LOGGER.debug(
            "voluptuous-openapi is not installed. LLM tool schema conversion "
            "will be limited. Install with: pip install voluptuous-openapi"
        )

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Render an image with Venice AI."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: AsyncVeniceAIClient = entry.runtime_data.client

        try:
            response = await client.images.generate(
                model="default",
                prompt=call.data["prompt"],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except VeniceAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        if not isinstance(response, dict):
            raise HomeAssistantError(
                f"Unexpected image API response type: {type(response).__name__}"
            )
        data = response.get("data", [{}])
        if not data or not isinstance(data, list) or len(data) < 1:
            raise HomeAssistantError("No image data returned from Venice AI")
        result = dict(data[0])
        result.pop("b64_json", None)
        return result

    # Only register AI Task service if platform is available
    if _HAS_AI_TASK:
        async def generate_data(call: ServiceCall) -> ServiceResponse:
            """Generate data using Venice AI Task."""
            entry_id = call.data["config_entry"]
            entry = hass.config_entries.async_get_entry(entry_id)

            if entry is None or entry.domain != DOMAIN:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="invalid_config_entry",
                    translation_placeholders={"config_entry": entry_id},
                )

            # Get the AI Task entity from runtime_data (Architecture 7.1 fix)
            ai_task_entity = entry.runtime_data.ai_task_entity

            if ai_task_entity is None:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="entity_not_found",
                    translation_placeholders={"entry_id": entry.entry_id},
                )

            task_text = call.data["task"]
            structure = call.data.get("structure")

            gen_task = ai_task.GenDataTask(
                task=task_text,
                structure=structure,
            )

            chat_log = conversation.ChatLog(
                conversation_id="service_call",
                content=[
                    conversation.UserContent(content=task_text)
                ]
            )

            try:
                result = await ai_task_entity._async_generate_data(gen_task, chat_log)
                return {
                    "conversation_id": result.conversation_id,
                    "data": result.data,
                }
            except Exception as err:
                raise HomeAssistantError(f"Error generating data: {err}") from err

        hass.services.async_register(
            DOMAIN,
            SERVICE_AI_TASK,
            generate_data,
            schema=vol.Schema(
                {
                    vol.Required("config_entry"): selector.ConfigEntrySelector(
                        {
                            "integration": DOMAIN,
                        }
                    ),
                    vol.Required("task"): cv.string,
                    vol.Optional("structure"): cv.string,
                }
            ),
            supports_response=SupportsResponse.ONLY,
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required("prompt"): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(
                    ("1024x1024", "1024x1792", "1792x1024")
                ),
                vol.Optional("quality", default="standard"): vol.In(("standard", "hd")),
                vol.Optional("style", default="vivid"): vol.In(("vivid", "natural")),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_repairs(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Set up the repairs platform for a config entry.

    Also registers a coordinator listener so that auth/API-availability repair
    issues are created/cleared automatically on every coordinator refresh,
    without any extra network calls (CRIT-2 fix).
    """
    try:
        from .repairs import (
            async_setup_entry as async_setup_repairs_entry,
            async_handle_coordinator_update,
        )

        await async_setup_repairs_entry(hass, entry)

        # Register coordinator listener — fires after every refresh cycle.
        # The listener is a no-arg callback; we close over hass, entry, and
        # the coordinator so repairs.async_handle_coordinator_update can inspect
        # coordinator.last_exception without making its own network calls.
        coordinator = entry.runtime_data.coordinator

        @callback
        def _on_coordinator_update() -> None:
            async_handle_coordinator_update(hass, entry, coordinator)

        entry.async_on_unload(coordinator.async_add_listener(_on_coordinator_update))
    except Exception:
        _LOGGER.debug("Repairs setup failed (likely unsupported HA version)")


async def async_unload_repairs(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Unload the repairs platform for a config entry."""
    try:
        from .repairs import async_unload_entry as async_unload_repairs_entry

        await async_unload_repairs_entry(hass, entry)
    except Exception:
        _LOGGER.debug("Repairs unload failed")


async def async_setup_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Set up Venice AI Conversation from a config entry."""
    client = AsyncVeniceAIClient(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )

    try:
        await client.models.list()
    except AuthenticationError as err:
        raise ConfigEntryAuthFailed("Invalid API key") from err
    except VeniceAIError as err:
        raise ConfigEntryNotReady(err) from err

    coordinator = VeniceAIDataUpdateCoordinator(hass, client)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = VeniceAIRuntimeData(
        client=client,
        coordinator=coordinator,
    )

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    entry.async_on_unload(client.close)

    _LOGGER.info("Forwarding entry setups to platforms: %s", PLATFORMS)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _LOGGER.info("Successfully forwarded entry setups")

    await async_setup_repairs(hass, entry)
    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload Venice AI when options change."""
    _LOGGER.info("Reloading Venice AI entry %s due to options update", entry.entry_id)
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Venice AI.

    Client cleanup is handled automatically via entry.async_on_unload(client.close)
    registered during async_setup_entry (HA best-practice — Architecture 7.1 / Item 19).
    """
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    await async_unload_repairs(hass, entry)
    return unload_ok
