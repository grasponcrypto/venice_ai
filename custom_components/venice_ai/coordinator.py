"""DataUpdateCoordinator for Venice AI."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .client import AsyncVeniceAIClient, AuthenticationError, VeniceAIError
from .const import UPDATE_INTERVAL

_LOGGER = logging.getLogger(__name__)


class VeniceAIDataUpdateCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to fetch and cache Venice AI metadata across platforms."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: AsyncVeniceAIClient,
    ) -> None:
        """Initialize the coordinator."""
        self.client = client
        super().__init__(
            hass,
            _LOGGER,
            name="Venice AI",
            update_interval=UPDATE_INTERVAL,
        )

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch models and voices from Venice AI.

        Each category is fetched independently so a failure in one
        does not block the others.
        """
        data: dict[str, Any] = {
            "text_models": [],
            "audio_models": [],
            "voices": [],
        }

        try:
            text_models = await self.client.models.list(model_type="text")
            if isinstance(text_models, list):
                data["text_models"] = text_models
                _LOGGER.debug("Coordinator fetched %d text models", len(text_models))
        except AuthenticationError as err:
            _LOGGER.error("Authentication error fetching text models: %s", err)
            raise UpdateFailed(f"Authentication error: {err}") from err
        except VeniceAIError as err:
            _LOGGER.warning("Error fetching text models: %s", err)
        except Exception:
            _LOGGER.exception("Unexpected error fetching text models")

        try:
            audio_models = await self.client.models.list(model_type="audio")
            if isinstance(audio_models, list):
                data["audio_models"] = audio_models
                _LOGGER.debug("Coordinator fetched %d audio models", len(audio_models))
        except AuthenticationError as err:
            _LOGGER.error("Authentication error fetching audio models: %s", err)
            raise UpdateFailed(f"Authentication error: {err}") from err
        except VeniceAIError as err:
            _LOGGER.warning("Error fetching audio models: %s", err)
        except Exception:
            _LOGGER.exception("Unexpected error fetching audio models")

        try:
            voices = await self.client.voices.list()
            if isinstance(voices, list):
                data["voices"] = voices
                _LOGGER.debug("Coordinator fetched %d voices", len(voices))
        except AuthenticationError as err:
            _LOGGER.error("Authentication error fetching voices: %s", err)
            raise UpdateFailed(f"Authentication error: {err}") from err
        except VeniceAIError as err:
            _LOGGER.warning("Error fetching voices: %s", err)
        except Exception:
            _LOGGER.exception("Unexpected error fetching voices")

        return data
