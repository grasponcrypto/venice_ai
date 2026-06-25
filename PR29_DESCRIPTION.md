## Summary

Consolidated rewrite of the Venice AI Home Assistant integration for the 0.9.x release line. All prior work for this PR has been squashed into a single, clean commit on `0.9-revise`.

## What's changed

### New platforms & components
- **STT** — Speech-to-Text platform using the Venice API.
- **TTS** — Text-to-Speech platform with dynamic model/voice selection and live config-entry options.
- **Coordinator** — `DataUpdateCoordinator` + `VeniceAIRuntimeData` for unified model/state management and reduced API calls.
- **Client** — New `AsyncVeniceAIClient` wrapper with typed errors (`RateLimitError`, `NetworkError`, `ServiceUnavailableError`), exponential-backoff retry logic, metrics, and diagnostics redaction.
- **Conversation** — Streaming chat support, tool-call fragment reassembly, `ChatLog` lifecycle fixes, LRU history eviction, and correct `async_added_to_hass` naming.
- **ai_task** service + sensor for generation tasks and usage metrics.
- **Diagnostics** — Redacted diagnostics downloader.

### Config flow & options
- Multi-step options flow with dynamic TTS voice population per model.
- `async_migrate_entry` support for future schema migrations.
- Re-auth flow for invalid/expired API keys.
- Shared Home Assistant HTTP session in options flow; no more leaked clients.

### Defaults & behavior
- Default chat model updated to `qwen-3-7-plus`.
- Streaming enabled by default.
- New options: `disable_thinking` and `strip_thinking_response`.
- Supported languages expanded from `[en]` to 10 languages.
- Empty STT audio handled gracefully.

### Reliability fixes
- Idempotent `AsyncVeniceAIClient.close()`.
- Correct `async_migrate_entry` signature.
- Removed invalid `repairs.py` platform.
- HACS manifest fix: `zip_release=false`.
- `ai_task` moved from `dependencies` to `after_dependencies`.
- Configurable tool-iteration limit.

### Documentation & metadata
- Added `CHANGELOG.md`, `CONTRIBUTING.md`, expanded `README.md`.
- Added `translations/en.json`, `icons.json`, `services.yaml`, `strings.json` updates.
- Added `pytest.ini` and a full pytest suite (`test_client.py`, `test_config_flow_tts.py`, `test_schema.py`, `test_venice_api.py`).

### Cleanup
- Removed stray `.cline/` plugin files, temporary text files, ad-hoc scripts, and now-untracked `CODE_REVIEW*.md` documents.
- Fixed all `ruff` F401 unused-import violations.

## Verification

- `python -m pytest tests` — **65 passed, 0 failed**
- `python -m ruff check custom_components/venice_ai tests` — **clean**

## Linked issues

- Closes #27
- Resolves #14, #15, #17, #18, #19, #20, #22, #24, #25, #26, #28
