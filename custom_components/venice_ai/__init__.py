"""The Venice AI Conversation integration."""

from __future__ import annotations

from .client import AsyncVeniceAIClient, VeniceAIError, AuthenticationError
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform, CONF_LLM_HASS_API
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
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict, 
    SelectSelector,
    SelectSelectorConfig,
)

from .const import DOMAIN, LOGGER

SERVICE_GENERATE_IMAGE = "generate_image"
PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

class VeniceAIConfigEntry(ConfigEntry):
    """Venice AI config entry with runtime data."""
    
    runtime_data: AsyncVeniceAIClient

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Venice AI Conversation."""
    LOGGER.debug("Setting up Venice AI integration")

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
                model="default",  # Venice AI uses 'default' model
                prompt=call.data["prompt"],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except VeniceAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        return response.data[0].model_dump(exclude={"b64_json"})

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

async def async_setup_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Set up Venice AI Conversation from a config entry."""
    LOGGER.debug("Setting up config entry: %s", entry.entry_id)
    client = AsyncVeniceAIClient(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
        base_url="https://api.venice.ai/api/v1"
    )

    try:
        await client.models.list()
    except AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except VeniceAIError as err:
        raise ConfigEntryNotReady(err) from err

    # Validate CONF_LLM_HASS_API
    llm_api_id = entry.options.get(CONF_LLM_HASS_API)
    if llm_api_id and (not isinstance(llm_api_id, str) or llm_api_id not in [api.id for api in llm.async_get_apis(hass)]):
        LOGGER.warning("Invalid CONF_LLM_HASS_API value '%s' in config entry; attempting to update to None", llm_api_id)
        try:
            result = hass.config_entries.async_update_entry(
                entry,
                options={**entry.options, CONF_LLM_HASS_API: None}
            )
            if isinstance(result, bool):
                LOGGER.debug("async_update_entry returned boolean: %s", result)
            else:
                await result  # Ensure the coroutine is awaited if it's not a boolean
            LOGGER.debug("Updated config entry options: %s", entry.options)
        except Exception as err:
            LOGGER.error("Failed to update CONF_LLM_HASS_API: %s; proceeding with setup", err)
    else:
        LOGGER.debug("CONF_LLM_HASS_API is valid or not set: %s", llm_api_id)

    # Store client in runtime_data
    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Venice AI."""
    LOGGER.debug("Unloading config entry: %s", entry.entry_id)
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

@callback
def async_get_options_schema(
    hass: HomeAssistant,
    options: dict[str, Any],
) -> vol.Schema:
    """Return the options schema."""
    apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control (disable entity control)",
            value="none",
        )
    ]
    apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    return vol.Schema(
        {
            vol.Optional(
                CONF_LLM_HASS_API,
                description={
                    "suggested_value": options.get(CONF_LLM_HASS_API, "none"),
                    "description": "Select the LLM API to use for controlling Home Assistant entities."
                },
                default="none",
            ): SelectSelector(SelectSelectorConfig(options=apis, mode=SelectSelectorMode.DROPDOWN)),
        }
    )