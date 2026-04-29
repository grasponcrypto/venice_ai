"""The Venice AI Conversation integration."""

from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.components import ai_task, conversation

from .client import AsyncVeniceAIClient, VeniceAIError, AuthenticationError
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

try:
    from voluptuous_openapi import convert as _voluptuous_convert  # noqa: F401
    _HAS_VOLUPTUOUS_OPENAPI = True
except ImportError:
    _HAS_VOLUPTUOUS_OPENAPI = False

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_AI_TASK = "ai_task"
PLATFORMS = (Platform.CONVERSATION, Platform.AI_TASK, Platform.TTS, Platform.STT)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


class VeniceAIConfigEntry(ConfigEntry):
    """Venice AI config entry with runtime data."""

    runtime_data: AsyncVeniceAIClient


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Venice AI Conversation."""
    if not _HAS_VOLUPTUOUS_OPENAPI:
        _LOGGER.warning(
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

        client: AsyncVeniceAIClient = entry.runtime_data

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

        # Response is a plain dict, not a Pydantic model
        data = response.get("data", [{}])[0]
        result = dict(data)
        result.pop("b64_json", None)
        return result

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

        # Get the AI Task entity from the integration's own data store
        ai_task_entity = hass.data.get(DOMAIN, {}).get(entry.entry_id)

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
    return True


async def async_setup_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Set up Venice AI Conversation from a config entry."""
    client = AsyncVeniceAIClient(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )

    try:
        await client.models.list()
    except AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except VeniceAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    _LOGGER.info("Forwarding entry setups to platforms: %s", PLATFORMS)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _LOGGER.info("Successfully forwarded entry setups")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Venice AI."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    # Remove our entity reference from the integration data store
    domain_data = hass.data.get(DOMAIN, {})
    domain_data.pop(entry.entry_id, None)
    client: AsyncVeniceAIClient = entry.runtime_data
    if client is not None:
        await client.close()
    return unload_ok
