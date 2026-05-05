"""Diagnostics support for Venice AI."""
from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN

# Fields redacted in full by async_redact_data (tokens, passwords, etc.).
# NOTE: "api_key" is intentionally excluded here — it is handled separately
# below to show the last 4 characters, which lets users identify which key
# is active during debugging without exposing the full secret.
TO_REDACT = {
    "token",
    "password",
    "secret",
    "authorization",
}


def _redact_api_key(value: str | None) -> str:
    """Return a partially-redacted API key showing only the last 4 characters.

    Example: ``"sk-abc123xyz"`` → ``"***xyz"``.
    Returns ``"***"`` if the value is empty or shorter than 4 characters.
    """
    if not value:
        return "***"
    return f"***{value[-4:]}" if len(value) >= 4 else "***"


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    runtime_data = entry.runtime_data
    client = runtime_data.client if runtime_data else None

    # Pre-process entry.data to apply custom api_key redaction before handing
    # the dict to async_redact_data (which would replace it entirely with
    # "**REDACTED**" if "api_key" were in TO_REDACT).
    entry_data = dict(entry.data)
    if "api_key" in entry_data:
        entry_data["api_key"] = _redact_api_key(entry_data["api_key"])

    entry_options = dict(entry.options)
    if "api_key" in entry_options:
        entry_options["api_key"] = _redact_api_key(entry_options["api_key"])

    diagnostics: dict[str, Any] = {
        "entry_id": entry.entry_id,
        "domain": entry.domain,
        "title": entry.title,
        "version": entry.version,
        "options": async_redact_data(entry_options, TO_REDACT),
        "data": async_redact_data(entry_data, TO_REDACT),
        "state": entry.state.value if hasattr(entry.state, "value") else str(entry.state),
        "client_available": client is not None,
    }

    return diagnostics
