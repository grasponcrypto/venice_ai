"""Keep speech entity names in step with the configured model."""

from __future__ import annotations

import logging
import re

# Names this integration generates. A user-chosen name that happens to use the
# same shape is released so a later model change is not stuck on the device page.
_GENERATED_SPEECH_NAME = re.compile(r"^(TTS|STT) \([^)]+\)$")
_LOGGER = logging.getLogger(__name__)


def registry_name_updates(
    original_name: str | None,
    custom_name: str | None,
    live_name: str,
) -> dict[str, str | None]:
    """Return entity-registry fields that still show an older model.

    Home Assistant keeps the first reported name. The device page then says
    ``TTS (tts-kokoro)`` after the options have moved on. ``original_name`` is
    what we last reported. ``name`` is a user override; clear it only when it
    is one of our generated names and no longer matches.
    """
    updates: dict[str, str | None] = {}
    if original_name != live_name:
        updates["original_name"] = live_name
    if (
        custom_name
        and custom_name != live_name
        and _GENERATED_SPEECH_NAME.fullmatch(custom_name)
    ):
        updates["name"] = None
    return updates


def publish_configured_name(entity: object) -> None:
    """Push the live name into the entity registry and the state machine."""
    hass = getattr(entity, "hass", None)
    entity_id = getattr(entity, "entity_id", None)
    if hass is None or not entity_id:
        return
    from homeassistant.helpers import entity_registry as er

    registry = er.async_get(hass)
    current = registry.async_get(entity_id)
    if current is not None:
        updates = registry_name_updates(
            getattr(current, "original_name", None),
            getattr(current, "name", None),
            entity.name,  # type: ignore[attr-defined]
        )
        if updates:
            try:
                registry.async_update_entity(entity_id, **updates)
            except TypeError:
                _LOGGER.debug(
                    "Could not refresh entity name for %s", entity_id, exc_info=True
                )
    write_state = getattr(entity, "async_write_ha_state", None)
    if write_state is not None:
        write_state()

