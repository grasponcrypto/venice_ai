"""Diagnostics support for Venice AI."""
from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import __version__ as HA_VERSION
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
    from . import VeniceAIRuntimeData

    runtime_data: VeniceAIRuntimeData | None = getattr(entry, "runtime_data", None)
    client = runtime_data.client if runtime_data else None
    coordinator = runtime_data.coordinator if runtime_data else None

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
        "homeassistant_version": HA_VERSION,
    }

    # Coordinator state (MIN-9 fix)
    if coordinator is not None:
        last_exception = None
        if coordinator.last_exception is not None:
            last_exception = f"{type(coordinator.last_exception).__name__}: {coordinator.last_exception}"

        coordinator_data = coordinator.data or {}
        text_models = coordinator_data.get("text_models", [])
        audio_models = coordinator_data.get("audio_models", [])
        voices = coordinator_data.get("voices", [])

        diagnostics["coordinator"] = {
            "last_update_success": coordinator.last_update_success,
            "last_exception": last_exception,
            "update_interval_seconds": coordinator.update_interval.total_seconds()
            if hasattr(coordinator, "update_interval") and coordinator.update_interval
            else None,
            "text_models_count": len(text_models),
            "audio_models_count": len(audio_models),
            "voices_count": len(voices),
        }
    else:
        diagnostics["coordinator"] = None

    return diagnostics
