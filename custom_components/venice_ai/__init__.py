"""The Venice AI Conversation integration."""

from __future__ import annotations

import asyncio
import logging
import uuid

import voluptuous as vol

from dataclasses import dataclass, field
from typing import Any


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
from homeassistant.helpers import config_validation as cv, issue_registry as ir, selector
from homeassistant.helpers.issue_registry import IssueSeverity
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

# Backwards-compatible import: older client.py may not define RateLimitError
try:
    from .client import RateLimitError
except ImportError:
    RateLimitError = None  # type: ignore[misc, assignment]
from .const import (
    CONF_CHAT_MODEL,
    CONF_TTS_MODEL,
    CONF_STT_MODEL,
    DOMAIN,
    HAS_VOLUPTUOUS_OPENAPI,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_STT_MODEL,
)
from .coordinator import VeniceAIDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_AI_TASK = "ai_task"
PLATFORMS = [Platform.CONVERSATION, Platform.TTS, Platform.STT, Platform.SENSOR]

if _HAS_AI_TASK:
    ai_task_platform = getattr(Platform, "AI_TASK", None)
    if ai_task_platform:
        PLATFORMS.append(ai_task_platform)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# ── Repair issue ID templates ────────────────────────────────────────────────
_ISSUE_DEPRECATED = "deprecated_model_{entry_id}_{model_key}"
_ISSUE_UNAVAIL = "unavailable_model_{entry_id}_{model_key}"
_ISSUE_AUTH = "auth_failure_{entry_id}"
_ISSUE_API_DOWN = "api_unavailable_{entry_id}"
_ISSUE_RATE_LIMIT = "rate_limited_{entry_id}"

# Map deprecated model IDs → recommended replacements (update as needed).
# Currently empty: no Venice AI models have been deprecated in the v1 API
# roster.  Populate this dict when Venice AI announces retirements, e.g.:
#   _DEPRECATED_MODELS = {"llama-2-70b": "llama-3.3-70b"}
# LOW-1: kept intentionally to preserve the model-check repair-issue loop;
# removing the dict would break the _async_create_model_issues code path.
_DEPRECATED_MODELS: dict[str, str] = {}


@dataclass
class VeniceAIRuntimeData:
    """Runtime data stored in the config entry."""

    client: AsyncVeniceAIClient
    coordinator: VeniceAIDataUpdateCoordinator
    ai_task_entity: object | None = None
    # HIGH-1: synchronization barrier signalled by the AI Task platform once its
    # entity has finished being added to HA. Consumers (e.g. the ai_task service)
    # await this before touching ``ai_task_entity`` to avoid a race where the
    # service fires before the entity exists.
    ai_task_ready: asyncio.Event = field(default_factory=asyncio.Event)



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

        if entry.runtime_data is None:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="config_entry_not_loaded",
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

            if entry.runtime_data is None:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="config_entry_not_loaded",
                    translation_placeholders={"config_entry": entry_id},
                )

            # HIGH-1: wait for the AI Task platform to finish adding its entity
            # before using it. async_forward_entry_setups returns once platforms
            # START loading, not once entities are registered, so a service call
            # fired immediately after setup could otherwise race ahead of the
            # entity. We bound the wait so a genuinely missing entity still fails
            # fast rather than hanging the service call indefinitely.
            ready: asyncio.Event = entry.runtime_data.ai_task_ready
            if not ready.is_set():
                try:
                    await asyncio.wait_for(ready.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    _LOGGER.warning(
                        "AI Task entity for entry %s was not ready within timeout",
                        entry.entry_id,
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
                instructions=task_text,
                structure=structure,
            )

            chat_log = conversation.ChatLog(
                conversation_id=str(uuid.uuid4()),
                content=[
                    conversation.UserContent(content=task_text)
                ]
            )

            try:
                # Call the public async_generate_data API instead of the
                # private _async_generate_data method (CRIT-3 fix).  This
                # respects the entity's public contract and avoids bypassing
                # any locking or lifecycle checks the platform may add.
                result = await ai_task_entity.async_generate_data(gen_task, chat_log)
                return {
                    "conversation_id": result.conversation_id,
                    "data": result.data,
                }
            except asyncio.CancelledError:
                raise
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


@callback
def _async_on_coordinator_update(
    hass: HomeAssistant, entry: ConfigEntry, coordinator: Any
) -> None:
    """Create or clear auth/API-availability repair issues based on coordinator state."""
    entry_id = entry.entry_id
    ir.async_delete_issue(hass, DOMAIN, _ISSUE_AUTH.format(entry_id=entry_id))
    ir.async_delete_issue(hass, DOMAIN, _ISSUE_API_DOWN.format(entry_id=entry_id))
    ir.async_delete_issue(hass, DOMAIN, _ISSUE_RATE_LIMIT.format(entry_id=entry_id))

    last_exc = coordinator.last_exception
    if last_exc is None:
        _LOGGER.debug(
            "Coordinator update succeeded for entry %s; connectivity issues cleared",
            entry_id,
        )
        return

    cause = getattr(last_exc, "__cause__", None) or last_exc

    if isinstance(cause, AuthenticationError):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _ISSUE_AUTH.format(entry_id=entry_id),
            is_fixable=False,
            is_persistent=True,
            severity=IssueSeverity.ERROR,
            translation_key="auth_failure",
            translation_placeholders={"entry_title": entry.title},
        )
        _LOGGER.warning(
            "Coordinator auth failure for entry %s — repair issue created", entry_id
        )
        # CRIT-1: programmatically start the reauth flow so HA opens the
        # re-authentication dialog without requiring the user to manually
        # locate and act on the repair issue.
        entry.async_start_reauth(hass)
    elif RateLimitError is not None and isinstance(cause, RateLimitError):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _ISSUE_RATE_LIMIT.format(entry_id=entry_id),
            is_fixable=False,
            is_persistent=False,
            severity=IssueSeverity.WARNING,
            translation_key="rate_limited",
            translation_placeholders={"entry_title": entry.title},
        )
        _LOGGER.warning(
            "Coordinator rate limit for entry %s — repair issue created", entry_id
        )
    else:
        ir.async_create_issue(
            hass,
            DOMAIN,
            _ISSUE_API_DOWN.format(entry_id=entry_id),
            is_fixable=False,
            is_persistent=False,
            severity=IssueSeverity.WARNING,
            translation_key="api_unavailable",
            translation_placeholders={
                "entry_title": entry.title,
                "error": str(cause),
            },
        )
        _LOGGER.warning(
            "Coordinator API error for entry %s — repair issue created: %s",
            entry_id,
            cause,
        )


async def _async_create_model_issues(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Check the Venice AI configuration and create repair issues as needed."""
    entry_id = entry.entry_id
    options = entry.options
    runtime_data = entry.runtime_data
    coordinator = getattr(runtime_data, "coordinator", None)

    available_text_models: set[str] = set()
    available_audio_models: set[str] = set()
    if coordinator and coordinator.data:
        available_text_models = {
            m.get("id", "")
            for m in coordinator.data.get("text_models", [])
            if isinstance(m, dict)
        }
        available_audio_models = {
            m.get("id", "")
            for m in coordinator.data.get("audio_models", [])
            if isinstance(m, dict)
        }
        _LOGGER.debug(
            "Repair check using coordinator data: %d text models, %d audio models",
            len(available_text_models),
            len(available_audio_models),
        )

    configured_models = {
        CONF_CHAT_MODEL: (
            options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            available_text_models,
        ),
        CONF_TTS_MODEL: (
            options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL),
            available_audio_models,
        ),
        CONF_STT_MODEL: (
            options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
            available_audio_models,
        ),
    }

    for model_key, (current_model, available_set) in configured_models.items():
        if current_model in _DEPRECATED_MODELS:
            issue_id = _ISSUE_DEPRECATED.format(
                entry_id=entry_id, model_key=model_key
            )
            ir.async_create_issue(
                hass,
                DOMAIN,
                issue_id,
                is_fixable=False,
                is_persistent=False,
                severity=IssueSeverity.WARNING,
                translation_key="deprecated_model",
                translation_placeholders={
                    "model": current_model,
                    "replacement": _DEPRECATED_MODELS[current_model],
                },
            )
            _LOGGER.debug(
                "Created repair issue for deprecated model %s in entry %s",
                current_model,
                entry_id,
            )
            continue

        if available_set and current_model not in available_set:
            issue_id = _ISSUE_UNAVAIL.format(
                entry_id=entry_id, model_key=model_key
            )
            ir.async_create_issue(
                hass,
                DOMAIN,
                issue_id,
                is_fixable=False,
                is_persistent=False,
                severity=IssueSeverity.ERROR,
                translation_key="unavailable_model",
                translation_placeholders={
                    "model": current_model,
                    "model_type": model_key.replace("_model", "").upper(),
                },
            )
            _LOGGER.warning(
                "Created repair issue for unavailable model %s (%s) in entry %s",
                current_model,
                model_key,
                entry_id,
            )


async def async_setup_repairs(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Set up repair issues for a config entry."""
    await _async_create_model_issues(hass, entry)

    coordinator = entry.runtime_data.coordinator

    @callback
    def _on_coordinator_update() -> None:
        _async_on_coordinator_update(hass, entry, coordinator)

    entry.async_on_unload(coordinator.async_add_listener(_on_coordinator_update))


async def async_unload_repairs(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Unload repair issues for a config entry."""
    entry_id = entry.entry_id
    registry = ir.async_get(hass)
    issues = [
        _ISSUE_DEPRECATED.format(entry_id=entry_id, model_key=CONF_CHAT_MODEL),
        _ISSUE_DEPRECATED.format(entry_id=entry_id, model_key=CONF_TTS_MODEL),
        _ISSUE_DEPRECATED.format(entry_id=entry_id, model_key=CONF_STT_MODEL),
        _ISSUE_UNAVAIL.format(entry_id=entry_id, model_key=CONF_CHAT_MODEL),
        _ISSUE_UNAVAIL.format(entry_id=entry_id, model_key=CONF_TTS_MODEL),
        _ISSUE_UNAVAIL.format(entry_id=entry_id, model_key=CONF_STT_MODEL),
        _ISSUE_AUTH.format(entry_id=entry_id),
        _ISSUE_API_DOWN.format(entry_id=entry_id),
        _ISSUE_RATE_LIMIT.format(entry_id=entry_id),
    ]
    for issue_id in issues:
        if registry.async_get_issue(DOMAIN, issue_id):
            ir.async_delete_issue(hass, DOMAIN, issue_id)
            _LOGGER.debug("Deleted repair issue %s", issue_id)


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

    _LOGGER.info("Forwarding entry setups to platforms: %s", PLATFORMS)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _LOGGER.info("Successfully forwarded entry setups")

    await async_setup_repairs(hass, entry)
    return True


async def async_reload_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> None:
    """Reload Venice AI when options change."""
    _LOGGER.info("Reloading Venice AI entry %s due to options update", entry.entry_id)
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """MAINT-2: migrate config entries to the current version.

    The ``version`` and ``minor_version`` keys on the config entry are
    inspected by HA to decide whether ``async_migrate_entry`` needs to run.
    Each migration should bump the version field once it completes so that
    the migration runs exactly once per upgrade.

    Currently the integration is at version 1 / minor 1, so this is the
    canonical entry point for future upgrade logic. Returning ``True``
    without bumping the version when there is nothing to do keeps the
    method in place as a stable extension point.
    """
    _LOGGER.debug(
        "Migrating Venice AI entry %s from version %s.%s to current version 1.1",
        entry.entry_id,
        entry.version,
        entry.minor_version,
    )
    # No data migration required yet; the dict-based storage layout has been
    # stable. Place future migrations (e.g. moving keys from data→options,
    # renaming CONF_* constants) inside this function.
    if entry.version > 1 or (entry.version == 1 and entry.minor_version >= 1):
        return True
    entry.version = 1
    entry.minor_version = 1
    return True


async def async_unload_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Unload Venice AI.

    Explicitly awaits client.close() after platforms are unloaded.
    Previously client.close was registered via entry.async_on_unload,
    but async_on_unload accepts only sync callables — an async close()
    would be called and its coroutine discarded, leaking the httpx
    session. By awaiting close() here we guarantee the client is shut
    down cleanly (CRIT-2 fix).
    """
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    await async_unload_repairs(hass, entry)

    # Explicitly close the async client — async_on_unload cannot await coroutines.
    client: AsyncVeniceAIClient = entry.runtime_data.client
    await client.close()

    return unload_ok
