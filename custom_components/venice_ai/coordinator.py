"""DataUpdateCoordinator for Venice AI."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .client import (
    AsyncVeniceAIClient,
    AuthenticationError,
    NetworkError,
    RateLimitError,
    ServiceUnavailableError,
    VeniceAIError,
)
from .const import UPDATE_INTERVAL

_LOGGER = logging.getLogger(__name__)


class VeniceAICoordinatorData(TypedDict):
    """Typed schema for Venice AI coordinator data (Architecture 7.2)."""

    text_models: list[dict[str, Any]]
    audio_models: list[dict[str, Any]]
    voices: list[str]


class VeniceAIDataUpdateCoordinator(DataUpdateCoordinator[VeniceAICoordinatorData]):
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

    async def _async_update_data(self) -> VeniceAICoordinatorData:
        """Fetch models and voices from Venice AI.

        Each category is fetched independently so a failure in one
        does not block the others.
        """
        data: VeniceAICoordinatorData = {
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
            raise UpdateFailed(f"Authentication failed: {err}") from err
        except RateLimitError as err:
            _LOGGER.warning("Rate limit exceeded fetching text models: %s", err)
            raise UpdateFailed(f"Rate limit exceeded: {err}") from err
        except ServiceUnavailableError as err:
            _LOGGER.warning("Venice AI service unavailable fetching text models: %s", err)
        except NetworkError as err:
            _LOGGER.warning("Network error fetching text models: %s", err)
        except VeniceAIError as err:
            _LOGGER.warning("Venice AI error fetching text models: %s", err)
        except Exception:
            _LOGGER.exception("Unexpected error fetching text models")

        try:
            tts_models = await self.client.models.list(model_type="tts")
            if isinstance(tts_models, list):
                for m in tts_models:
                    m["model_type"] = "tts"
                data["audio_models"].extend(tts_models)
                _LOGGER.debug("Coordinator fetched %d TTS models", len(tts_models))
                # Extract voices from TTS model metadata (voice_models field per model)
                for model in tts_models:
                    voice_models = model.get("voice_models", [])
                    if isinstance(voice_models, list):
                        for vid in voice_models:
                            if isinstance(vid, str) and vid not in data["voices"]:
                                data["voices"].append(vid)
        except AuthenticationError as err:
            _LOGGER.error("Authentication error fetching TTS models: %s", err)
            raise UpdateFailed(f"Authentication failed: {err}") from err
        except RateLimitError as err:
            _LOGGER.warning("Rate limit exceeded fetching TTS models: %s", err)
            raise UpdateFailed(f"Rate limit exceeded: {err}") from err
        except ServiceUnavailableError as err:
            _LOGGER.warning("Venice AI service unavailable fetching TTS models: %s", err)
        except NetworkError as err:
            _LOGGER.warning("Network error fetching TTS models: %s", err)
        except VeniceAIError as err:
            _LOGGER.warning("Venice AI error fetching TTS models: %s", err)
        except Exception:
            _LOGGER.exception("Unexpected error fetching TTS models")

        try:
            asr_models = await self.client.models.list(model_type="asr")
            if isinstance(asr_models, list):
                for m in asr_models:
                    m["model_type"] = "asr"
                data["audio_models"].extend(asr_models)
                _LOGGER.debug("Coordinator fetched %d ASR models", len(asr_models))
        except AuthenticationError as err:
            _LOGGER.error("Authentication error fetching ASR models: %s", err)
            raise UpdateFailed(f"Authentication failed: {err}") from err
        except RateLimitError as err:
            _LOGGER.warning("Rate limit exceeded fetching ASR models: %s", err)
            raise UpdateFailed(f"Rate limit exceeded: {err}") from err
        except ServiceUnavailableError as err:
            _LOGGER.warning("Venice AI service unavailable fetching ASR models: %s", err)
        except NetworkError as err:
            _LOGGER.warning("Network error fetching ASR models: %s", err)
        except VeniceAIError as err:
            _LOGGER.warning("Venice AI error fetching ASR models: %s", err)
        except Exception:
            _LOGGER.exception("Unexpected error fetching ASR models")

        # MED-1: If every fetch failed and we have no data at all, surface the
        # failure to the coordinator so HA can apply back-off, show the entity
        # as unavailable, and fire repair issues.  Partial failures (e.g. only
        # ASR models missing) are tolerated so other platforms keep working.
        if not data["text_models"] and not data["audio_models"] and not data["voices"]:
            raise UpdateFailed(
                "All Venice AI data fetches failed; coordinator has no data to return."
            )

        return data
