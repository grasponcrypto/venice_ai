"""Diagnostic sensors for Venice AI usage metrics (LOW-4).

Exposes per-config-entry usage telemetry — request counts, error counts, and
token consumption — as diagnostic sensor entities so users can monitor API
usage without enabling debug logging. The counters are sourced from the
``VeniceAIMetrics`` instance carried on the shared API client.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import VeniceAIMetrics
from .const import DOMAIN

if TYPE_CHECKING:
    from . import VeniceAIRuntimeData


@dataclass(frozen=True, kw_only=True)
class VeniceAISensorDescription(SensorEntityDescription):
    """Describes a Venice AI diagnostic sensor."""

    value_fn: Callable[[VeniceAIMetrics], int | str | None]


SENSORS: tuple[VeniceAISensorDescription, ...] = (
    VeniceAISensorDescription(
        key="request_count",
        translation_key="request_count",
        name="API requests",
        icon="mdi:api",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.request_count,
    ),
    VeniceAISensorDescription(
        key="error_count",
        translation_key="error_count",
        name="API errors",
        icon="mdi:alert-circle",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.error_count,
    ),
    VeniceAISensorDescription(
        key="total_tokens",
        translation_key="total_tokens",
        name="Total tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.total_tokens,
    ),
    VeniceAISensorDescription(
        key="prompt_tokens",
        translation_key="prompt_tokens",
        name="Prompt tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.prompt_tokens,
    ),
    VeniceAISensorDescription(
        key="completion_tokens",
        translation_key="completion_tokens",
        name="Completion tokens",
        icon="mdi:counter",
        native_unit_of_measurement="tokens",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.completion_tokens,
    ),
    VeniceAISensorDescription(
        key="last_error",
        translation_key="last_error",
        name="Last error",
        icon="mdi:message-alert",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda m: m.last_error,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Venice AI diagnostic sensors from a config entry."""
    runtime_data: VeniceAIRuntimeData = entry.runtime_data
    async_add_entities(
        VeniceAIUsageSensor(entry, runtime_data, description)
        for description in SENSORS
    )


class VeniceAIUsageSensor(SensorEntity):
    """A diagnostic sensor reporting a single Venice AI usage metric."""

    _attr_has_entity_name = True
    _attr_should_poll = True
    entity_description: VeniceAISensorDescription

    def __init__(
        self,
        entry: ConfigEntry,
        runtime_data: "VeniceAIRuntimeData",
        description: VeniceAISensorDescription,
    ) -> None:
        """Initialize the usage sensor."""
        self.entity_description = description
        self._metrics = runtime_data.client.metrics
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
        )

    @property
    def native_value(self) -> int | str | None:
        """Return the current metric value."""
        return self.entity_description.value_fn(self._metrics)
