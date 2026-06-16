# Changelog

All notable changes to **Venice AI for Home Assistant** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- **CRIT-1** — `async_start_reauth()` is now called automatically when the coordinator detects an `AuthenticationError`, opening the re-authentication dialog without requiring manual user action on the repair issue.
- **HIGH-2** — Periodic conversation cleanup task in `VeniceAIConversationEntity`. Conversations that fill the LRU cache are now evicted on a schedule (configurable via `CONVERSATION_TTL_SECONDS`) rather than only on access. Prevents memory growth on long-lived HA instances.
- **HIGH-4** — `CONF_REQUEST_TIMEOUT` option exposed in the options flow UI, allowing users to raise the per-request timeout for slow connections or large payloads (default: 60 s).
- **MED-1** — `coordinator.py` now raises `UpdateFailed` when all three data fetches (text models, TTS models, ASR models) fail, surfacing the failure to Home Assistant's coordinator machinery rather than silently returning empty data.
- **MED-2** — Template caching in `VeniceAIConversationEntity`. Compiled `Template` objects are cached by their source string and reused across turns, avoiding re-parsing on every conversation request.
- **MED-3** — Opt-in streaming mode for conversation responses (`CONF_STREAM_RESPONSE`). When enabled, the integration consumes the Venice AI streaming chat API and reassembles deltas before returning the final response.
- **LOW-1** — `_DEPRECATED_MODELS` dict in `__init__.py` documented with an explicit comment. The dict is intentionally kept empty until Venice AI retires models; the repair-issue code path is preserved.
- **LOW-2** — `async_reload_entry` and `async_unload_entry` in `__init__.py` typed with `VeniceAIConfigEntry` (typed subclass of `ConfigEntry`) for stronger static-analysis coverage.
- **LOW-3** — `extra_state_attributes` property added to `VeniceAIConversationEntity`, exposing `active_conversations` count as a HA state attribute for dashboard visibility.
- **PERF-2** — Options flow (`config_flow.py`) now reuses live coordinator data when available, avoiding a redundant API round-trip every time the options flow is opened.
- **DOC-3** — `CONTRIBUTING.md` added with development setup, coding standards, test instructions, and contribution guidelines.
- **DOC-4** — `CHANGELOG.md` introduced (this file).

### Changed
- `_fetch_model_options` in `VeniceAIOptionsFlow` now short-circuits with coordinator data before falling back to a fresh API call (PERF-2).
- `async_added_to_hass` in `VeniceAIConversationEntity` now schedules a periodic background cleanup task (HIGH-2).

### Fixed
- **CRIT-1** — Authentication failures detected by the coordinator now automatically trigger re-authentication without user having to find the repair issue manually.
- **MED-1** — Silent empty-data returns from the coordinator on total fetch failure are replaced with a proper `UpdateFailed` exception.

---

## [1.0.0] — Initial Release

### Added
- Full `ConversationEntity` implementation with multi-turn chat history (LRU, max 20 conversations).
- Text-to-Speech platform (`tts.py`) using Venice AI TTS API.
- Speech-to-Text platform (`stt.py`) using Venice AI ASR API.
- `DataUpdateCoordinator` for periodic model/voice metadata refresh (hourly).
- Diagnostic `sensor.py` platform exposing model counts.
- `ai_task.py` platform for HA 2024.8+ AI Task support.
- Config flow with API key validation and re-authentication support.
- Options flow with live model/voice dropdowns, all tunable parameters.
- Tool/function-calling support via HA LLM API integration.
- Streaming mode (opt-in via options).
- Repair issue creation for authentication failures, rate limiting, deprecated/unavailable models.
- `CONTRIBUTING.md` and `CHANGELOG.md` documentation.
