"""The Venice AI Conversation integration."""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid

import voluptuous as vol

from dataclasses import dataclass, field
from typing import Any


from homeassistant.config_entries import ConfigEntry, ConfigEntryState
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

# Conditional import for ai_task (availability depends on HA version).
# Bind it under another name. Loading this integration's ai_task platform
# sets custom_components.venice_ai.ai_task, which would replace a binding
# named ai_task. The service would then look up GenDataTask on the platform
# module and crash with "has no attribute 'GenDataTask'".
try:
    from homeassistant.components import ai_task as ha_ai_task

    _HAS_AI_TASK = True
except ImportError:
    ha_ai_task = None  # type: ignore[assignment]
    _HAS_AI_TASK = False

from .ai_task_support import build_gen_data_task
from .client import (
    IMAGE_GENERATE_FORMAT,
    AsyncVeniceAIClient,
    AuthenticationError,
    ImageGenerateError,
    RateLimitError,
    VeniceAIError,
    find_image_model,
    image_model_constraints,
)
from .const import (
    CONF_CHAT_MODEL,
    CONF_TTS_MODEL,
    CONF_STT_MODEL,
    DOMAIN,
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

# Retired model IDs → recommended replacements. Existing entries still using
# a retired ID get a repair issue instead of failing tool calls with no hint.
_DEPRECATED_MODELS: dict[str, str] = {
    "e2ee-gemma-4-31b": RECOMMENDED_CHAT_MODEL,
}


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


async def _async_dispatch_ai_task(
    hass: HomeAssistant,
    entity: object,
    gen_task: object,
    task_text: str,
    context: Any,
) -> Any:
    """Run a data task on the AI Task API this core actually provides.

    2026.10 exposes ``internal_async_generate_data`` and builds the chat log
    itself. Older cores still expect ``async_generate_data(task, chat_log)``.
    """
    internal = getattr(entity, "internal_async_generate_data", None)
    if callable(internal):
        from homeassistant.helpers.chat_session import async_get_chat_session

        with async_get_chat_session(hass) as session:
            if "context" in inspect.signature(internal).parameters:
                return await internal(session, gen_task, context)
            return await internal(session, gen_task)

    chat_log = conversation.ChatLog(
        conversation_id=str(uuid.uuid4()),
        content=[conversation.UserContent(content=task_text)],
    )
    return await entity.async_generate_data(gen_task, chat_log)  # type: ignore[attr-defined]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Venice AI Conversation.

    Services are registered from ``async_setup_entry`` so they cannot be
    called before a config entry has loaded.
    """
    return True


def _async_register_services(hass: HomeAssistant) -> None:
    """Register services once, when the first config entry loads."""
    if hass.services.has_service(DOMAIN, SERVICE_GENERATE_IMAGE):
        return

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
        model_id = call.data["model"]
        coordinator = entry.runtime_data.coordinator
        image_models = []
        if coordinator is not None and coordinator.data:
            image_models = coordinator.data.get("image_models") or []
        if not isinstance(image_models, list) or not image_models:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="image_models_unavailable",
            )
        model = find_image_model(image_models, model_id)
        if model is None:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="unknown_image_model",
                translation_placeholders={"model": model_id},
            )

        try:
            response = await client.images.generate(
                model_id,
                call.data["prompt"],
                constraints=image_model_constraints(model),
                aspect_ratio=call.data.get("aspect_ratio"),
                resolution=call.data.get("resolution"),
                width=call.data.get("width"),
                height=call.data.get("height"),
            )
        except ImageGenerateError as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_image_request",
                translation_placeholders={"reason": str(err)},
            ) from err
        except VeniceAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        if not isinstance(response, dict):
            raise HomeAssistantError(
                f"Unexpected image API response type: {type(response).__name__}"
            )
        images = response.get("images")
        if (
            not isinstance(images, list)
            or not images
            or not isinstance(images[0], str)
            or not images[0]
        ):
            if response.get("content_policy_violation_count"):
                raise HomeAssistantError(
                    "Venice AI rejected the image under its content policy"
                )
            raise HomeAssistantError("No image data returned from Venice AI")
        return {
            "id": response.get("id"),
            "model": model_id,
            "format": IMAGE_GENERATE_FORMAT,
            "image": images[0],
        }

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

            try:
                # Build the task inside the handler. On 2026.10 GenDataTask
                # requires ``name``; constructing it above this try raised
                # TypeError, which Developer Tools reports as Unknown error
                # and which never reached record_error.
                # ha_ai_task is the core module, not this integration's platform.
                if ha_ai_task is None:
                    raise HomeAssistantError("AI Task is not available")
                gen_task = build_gen_data_task(
                    ha_ai_task.GenDataTask,
                    task_text,
                    structure,
                )
                result = await _async_dispatch_ai_task(
                    hass,
                    ai_task_entity,
                    gen_task,
                    task_text,
                    call.context,
                )
                return {
                    "conversation_id": result.conversation_id,
                    "data": result.data,
                }
            except asyncio.CancelledError:
                raise
            except HomeAssistantError:
                raise
            except Exception as err:
                metrics = getattr(
                    getattr(entry.runtime_data, "client", None), "metrics", None
                )
                if metrics is not None:
                    metrics.record_error(err)
                _LOGGER.exception("Venice AI task failed")
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
                vol.Required("model"): cv.string,
                vol.Optional("aspect_ratio"): cv.string,
                vol.Optional("resolution"): cv.string,
                vol.Optional("width"): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=1280)
                ),
                vol.Optional("height"): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=1280)
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )


def _async_unregister_services(hass: HomeAssistant) -> None:
    """Remove services when the last config entry unloads."""
    if hass.services.has_service(DOMAIN, SERVICE_GENERATE_IMAGE):
        hass.services.async_remove(DOMAIN, SERVICE_GENERATE_IMAGE)
    if hass.services.has_service(DOMAIN, SERVICE_AI_TASK):
        hass.services.async_remove(DOMAIN, SERVICE_AI_TASK)


def _has_other_loaded_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Return True if another Venice AI entry is still loaded."""
    return any(
        other.entry_id != entry.entry_id and other.state is ConfigEntryState.LOADED
        for other in hass.config_entries.async_entries(DOMAIN)
    )


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
    elif isinstance(cause, RateLimitError):
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
        await client.models.list(force_refresh=True)
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

    # NOTE: No manual add_update_listener is needed here.  VeniceAIOptionsFlow
    # subclasses OptionsFlowWithReload (HA ≥ 2024.1) which automatically
    # triggers an integration reload when the user saves options.

    _LOGGER.info("Forwarding entry setups to platforms: %s", PLATFORMS)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _LOGGER.info("Successfully forwarded entry setups")

    await async_setup_repairs(hass, entry)
    _async_register_services(hass)
    return True


# NOTE: async_reload_entry is intentionally NOT defined.
# VeniceAIOptionsFlow subclasses OptionsFlowWithReload (HA ≥ 2024.1) which
# automatically triggers an integration reload when the user saves options.
# Defining async_reload_entry would register it as an update listener,
# which conflicts with OptionsFlowWithReload and raises:
#   ValueError: Config entry update listeners should not be used with OptionsFlowWithReload


async def async_migrate_entry(hass: HomeAssistant, entry: VeniceAIConfigEntry) -> bool:
    """Migrate config entries to the current version (1.1).

    Home Assistant calls this when the stored version does not match the
    flow's ``VERSION`` / ``MINOR_VERSION``. A newer major version means the
    install was downgraded; returning ``False`` refuses to load an entry this
    code does not understand. Bumps go through ``async_update_entry`` so they
    are persisted.
    """
    _LOGGER.debug(
        "Migrating Venice AI entry %s from version %s.%s",
        entry.entry_id,
        entry.version,
        entry.minor_version,
    )

    if entry.version > 1:
        return False

    if entry.version < 1:
        hass.config_entries.async_update_entry(entry, version=1, minor_version=1)
        return True

    if entry.version == 1 and entry.minor_version < 1:
        # 1.0 -> 1.1. The stored layout is unchanged.
        hass.config_entries.async_update_entry(entry, minor_version=1)

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

    if unload_ok and not _has_other_loaded_entry(hass, entry):
        _async_unregister_services(hass)

    return unload_ok
