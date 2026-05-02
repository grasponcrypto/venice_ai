"""Diagnostics support for Venice AI."""
from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN

TO_REDACT = {
    "api_key",
    "token",
    "password",
    "secret",
    "authorization",
}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    runtime_data = entry.runtime_data
    client = runtime_data.client if runtime_data else None

    diagnostics: dict[str, Any] = {
        "entry_id": entry.entry_id,
        "domain": entry.domain,
        "title": entry.title,
        "version": entry.version,
        "options": async_redact_data(dict(entry.options), TO_REDACT),
        "data": async_redact_data(dict(entry.data), TO_REDACT),
        "state": entry.state.value if hasattr(entry.state, "value") else str(entry.state),
        "client_available": client is not None,
    }

    return diagnostics
