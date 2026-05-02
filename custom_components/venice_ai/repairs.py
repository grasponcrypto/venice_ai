"""Repairs platform for Venice AI.

Creates and manages Home Assistant repair issues for the Venice AI integration,
surfacing non-fatal problems (deprecated models, API availability, etc.) in
the Settings → Repairs UI.
"""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.issue_registry import IssueSeverity

from .client import AsyncVeniceAIClient, AuthenticationError, VeniceAIError
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
ISSUE_ID_AUTH_FAILURE = "auth_failure_{entry_id}"
ISSUE_ID_API_UNAVAILABLE = "api_unavailable_{entry_id}"


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

    # 1. Deprecated model checks
    deprecated_models = {
        CONF_CHAT_MODEL: options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
        CONF_TTS_MODEL: options.get(CONF_TTS_MODEL, RECOMMENDED_TTS_MODEL),
        CONF_STT_MODEL: options.get(CONF_STT_MODEL, RECOMMENDED_STT_MODEL),
    }
    for model_key, current_model in deprecated_models.items():
        # Expand this set as Venice AI deprecates models over time
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

    # 2. API connectivity / auth check
    api_key = entry.data.get("api_key")
    if api_key:
        try:
            async with AsyncVeniceAIClient(api_key=api_key) as client:
                await client.models.list()
        except AuthenticationError:
            issue_id = ISSUE_ID_AUTH_FAILURE.format(entry_id=entry_id)
            ir.async_create_issue(
                hass,
                DOMAIN,
                issue_id,
                is_fixable=True,
                is_persistent=True,
                severity=IssueSeverity.ERROR,
                translation_key="auth_failure",
                translation_placeholders={"entry_title": entry.title},
            )
            _LOGGER.warning(
                "Created repair issue for auth failure in entry %s", entry_id
            )
        except VeniceAIError as err:
            issue_id = ISSUE_ID_API_UNAVAILABLE.format(entry_id=entry_id)
            ir.async_create_issue(
                hass,
                DOMAIN,
                issue_id,
                is_fixable=False,
                is_persistent=False,
                severity=IssueSeverity.WARNING,
                translation_key="api_unavailable",
                translation_placeholders={
                    "entry_title": entry.title,
                    "error": str(err),
                },
            )
            _LOGGER.warning(
                "Created repair issue for API unavailability in entry %s: %s",
                entry_id,
                err,
            )
        except Exception:
            _LOGGER.exception(
                "Unexpected error during repair check for entry %s", entry_id
            )


@callback
def _async_delete_entry_issues(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Delete all repair issues tied to a config entry."""
    entry_id = entry.entry_id
    registry = ir.async_get(hass)
    issues = [
        ISSUE_ID_DEPRECATED_MODEL.format(entry_id=entry_id, model_key=CONF_CHAT_MODEL),
        ISSUE_ID_DEPRECATED_MODEL.format(entry_id=entry_id, model_key=CONF_TTS_MODEL),
        ISSUE_ID_DEPRECATED_MODEL.format(entry_id=entry_id, model_key=CONF_STT_MODEL),
        ISSUE_ID_AUTH_FAILURE.format(entry_id=entry_id),
        ISSUE_ID_API_UNAVAILABLE.format(entry_id=entry_id),
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
