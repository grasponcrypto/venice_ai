"""Repairs platform for Venice AI.

Creates and manages Home Assistant repair issues for the Venice AI integration,
surfacing non-fatal problems (deprecated models, API availability, etc.) in
the Settings → Repairs UI.

Design note (CRIT-2 fix):
    This module must NOT make its own network calls.  Auth and API-availability
    issues are now surfaced exclusively via ``async_handle_coordinator_update``,
    which is registered as a coordinator listener in ``__init__.py`` and fires
    every time the coordinator refreshes (successfully or with an error).  The
    former ``async with AsyncVeniceAIClient(...)`` block created a redundant
    HTTP session on every HA restart / config-entry reload and could leak the
    underlying ``aiohttp.ClientSession`` if ``__aexit__`` was not reached due
    to an exception inside the context manager.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.issue_registry import IssueSeverity

from .client import AuthenticationError, RateLimitError
from .const import (
    CONF_CHAT_MODEL,
    CONF_TTS_MODEL,
    CONF_STT_MODEL,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_TTS_MODEL,
    RECOMMENDED_STT_MODEL,
)

_LOGGER = logging.getLogger(__name__)

ISSUE_ID_DEPRECATED_MODEL = "deprecated_model_{entry_id}_{model_key}"
ISSUE_ID_UNAVAIL_MODEL = "unavailable_model_{entry_id}_{model_key}"
ISSUE_ID_AUTH_FAILURE = "auth_failure_{entry_id}"
ISSUE_ID_API_UNAVAILABLE = "api_unavailable_{entry_id}"
ISSUE_ID_RATE_LIMITED = "rate_limited_{entry_id}"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
) -> bool:
    """Set up Venice AI repairs for a config entry."""
    await _async_check_and_create_issues(hass, config_entry)
    return True


async def async_unload_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
) -> bool:
    """Clean up repair issues for a config entry."""
    _async_delete_entry_issues(hass, config_entry)
    return True


async def _async_check_and_create_issues(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Check the Venice AI configuration and create repair issues as needed."""
    entry_id = entry.entry_id
    options = entry.options
    runtime_data = entry.runtime_data
    coordinator = getattr(runtime_data, "coordinator", None)

    # Gather available models from coordinator if loaded, otherwise skip dynamic check
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
        # 1. Static deprecation check
        if current_model in _DEPRECATED_MODELS:
            issue_id = ISSUE_ID_DEPRECATED_MODEL.format(
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

        # 2. Dynamic unavailable model check (only when coordinator data exists)
        if available_set and current_model not in available_set:
            issue_id = ISSUE_ID_UNAVAIL_MODEL.format(
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

    # NOTE: Auth / API-availability issues are intentionally NOT checked here via a
    # direct network call (CRIT-2 fix).  They are handled exclusively by
    # ``async_handle_coordinator_update`` which piggybacks on the coordinator's
    # existing refresh cycle.


@callback
def async_handle_coordinator_update(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator: Any,
) -> None:
    """Create or clear auth/API-availability repair issues based on coordinator state.

    This callback is registered as a ``DataUpdateCoordinator`` listener in
    ``__init__.py`` so it fires automatically after every scheduled or manual
    refresh — no independent network calls are made here.

    On success  → all connectivity issues are cleared.
    On auth failure (``AuthenticationError`` cause) → ``auth_failure`` issue raised.
    On other API error  → ``api_unavailable`` issue raised.
    """
    entry_id = entry.entry_id

    # Always clear previous connectivity issues first; they will be recreated
    # below if the failure is still present.
    ir.async_delete_issue(hass, DOMAIN, ISSUE_ID_AUTH_FAILURE.format(entry_id=entry_id))
    ir.async_delete_issue(hass, DOMAIN, ISSUE_ID_API_UNAVAILABLE.format(entry_id=entry_id))
    ir.async_delete_issue(hass, DOMAIN, ISSUE_ID_RATE_LIMITED.format(entry_id=entry_id))

    last_exc = coordinator.last_exception
    if last_exc is None:
        _LOGGER.debug(
            "Coordinator update succeeded for entry %s; connectivity issues cleared",
            entry_id,
        )
        return

    # Unwrap UpdateFailed to find the root cause exception.
    cause = getattr(last_exc, "__cause__", None) or last_exc

    if isinstance(cause, AuthenticationError):
        ir.async_create_issue(
            hass,
            DOMAIN,
            ISSUE_ID_AUTH_FAILURE.format(entry_id=entry_id),
            is_fixable=True,
            is_persistent=True,
            severity=IssueSeverity.ERROR,
            translation_key="auth_failure",
            translation_placeholders={"entry_title": entry.title},
        )
        _LOGGER.warning(
            "Coordinator auth failure for entry %s — repair issue created", entry_id
        )
    elif isinstance(cause, RateLimitError):
        ir.async_create_issue(
            hass,
            DOMAIN,
            ISSUE_ID_RATE_LIMITED.format(entry_id=entry_id),
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
            ISSUE_ID_API_UNAVAILABLE.format(entry_id=entry_id),
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


@callback
def _async_delete_entry_issues(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Delete all repair issues tied to a config entry."""
    entry_id = entry.entry_id
    registry = ir.async_get(hass)
    issues = [
        ISSUE_ID_DEPRECATED_MODEL.format(
            entry_id=entry_id, model_key=CONF_CHAT_MODEL
        ),
        ISSUE_ID_DEPRECATED_MODEL.format(
            entry_id=entry_id, model_key=CONF_TTS_MODEL
        ),
        ISSUE_ID_DEPRECATED_MODEL.format(
            entry_id=entry_id, model_key=CONF_STT_MODEL
        ),
        ISSUE_ID_UNAVAIL_MODEL.format(entry_id=entry_id, model_key=CONF_CHAT_MODEL),
        ISSUE_ID_UNAVAIL_MODEL.format(entry_id=entry_id, model_key=CONF_TTS_MODEL),
        ISSUE_ID_UNAVAIL_MODEL.format(entry_id=entry_id, model_key=CONF_STT_MODEL),
        ISSUE_ID_AUTH_FAILURE.format(entry_id=entry_id),
        ISSUE_ID_API_UNAVAILABLE.format(entry_id=entry_id),
        ISSUE_ID_RATE_LIMITED.format(entry_id=entry_id),
    ]
    for issue_id in issues:
        if registry.async_get_issue(DOMAIN, issue_id):
            ir.async_delete_issue(hass, DOMAIN, issue_id)
            _LOGGER.debug("Deleted repair issue %s", issue_id)


# Map deprecated model IDs to their recommended replacements.
# Update this dict as Venice AI deprecates models.
_DEPRECATED_MODELS: dict[str, str] = {
    # Example: "old-model-id": "new-model-id",
}
