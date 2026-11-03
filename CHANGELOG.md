# Changelog

All notable changes to **Venice AI Conversation** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

### Changed
- Default chat model is now `deepseek-v4-1-flash-1017-uncensored`, the model Venice marks as `default` and `function_calling_default`. `e2ee-gemma-4-31b` is retired and raises a repair pointing at the new default.

### Fixed
- Config and re-auth flows pass Home Assistant's shared HTTP client, so certificate loading does not block the event loop.
- Options copy and the README match the unfiltered text-model list and the actions this branch actually registers.

## [1.1.0-dev.3] - 2026-11-02

### Fixed
- **STT state:** Home Assistant 2026.10 owns speech-to-text state and stores the last processed time there. Showing the model id in that state made the UI say Invalid timestamp after a successful transcription. The model is now an entity attribute. The state is left to Home Assistant.
- **AI Task action:** `venice_ai.ai_task` built `GenDataTask` without the required `name` and built `ChatLog` with the old constructor. On 2026.10 that raised `TypeError` before the handler could record it, so Developer Tools showed Unknown error and Last error stayed `none`. The action now uses `internal_async_generate_data` when that API exists.

## [1.1.0-dev.2] - 2026-11-02

### Fixed
- **Generate Image action:** The Actions form had no inputs, so a click failed with `required key not provided` for `config_entry` and `prompt`. Field selectors now live in `services.yaml`.
- **STT row:** The speech-to-text entity is named STT and shows the configured model instead of Unknown.
- **Last error:** An empty sensor state is still Home Assistant's Unknown. When nothing has failed, the sensor now reads `none`.

## [1.1.0-dev.1] - 2026-11-02

### Fixed
- **#36:** Streaming chat, TTS, STT, and image generation now record diagnostic metrics. Previously only non-streaming chat did, so the sensors stayed at 0 while Stream response was on (the default). Last error shows blank when nothing has failed, instead of Home Assistant's "Unknown".
- **Conversation entity:** Adopt the pre-0.9 agent (`unique_id` was the config entry id, name "VeniceAI Conversation") so Assist pipelines keep using it. If a newer agent already exists, pipelines that still name the old entity are pointed at the live one and the orphan is removed.

## [1.1.0-dev] - 2026-11-01

### Added
- **DOC-1:** Module-level docstrings on all public client classes
  (`AsyncVeniceAIClient`, `ChatCompletions`, `Models`, `Speech`,
  `Transcriptions`, `Images`, `VeniceAIMetrics`).
- **DOC-2:** README **Operations** and **Troubleshooting** sections.
- **MAINT-3:** `FEATURE_MIN_VERSIONS` table in `const.py` referencing
  HA minimum versions for `ai_task`, `streaming_tts`,
  `conversation_entity`, and `sensor_total_increasing`.
- **MED-4:** Retry constants `MAX_RETRIES`, `RETRY_BASE_DELAY`,
  `RETRY_MAX_DELAY` extracted to `const.py`.
- **PERF-4:** `httpx.Limits` configured with `DEFAULT_HTTP_KEEPALIVE`
  and `DEFAULT_HTTP_MAX_CONNECTIONS` from `const.py`.
- **SEC-1:** API-key redaction helper (`_redact_api_key`) used by
  `diagnostics.py`, plus header-value sanitiser
  (`_sanitize_header_value`) in `client.py` to scrub CR/LF.
- **SEC-2:** Tool-call argument validation in `conversation.py`
  ensures tool args are JSON objects before invocation.
- **TEST-3:** `tests/test_schema.py` adds 16 schema-conversion tests
  covering `_format_venice_schema` and `_convert_schema_to_hashable`.
- **MAINT-2:** `async_migrate_entry` migrates older entries to the
  current `(version=1, minor_version=1)` schema.

### Changed
- **#46:** `generate_image` calls native `POST /image/generate` and requires an image model id from the coordinator's `type=image` list. Optional `aspect_ratio`, `resolution`, `width`, and `height` are sent only when that model's constraints list them. The response is `{id, model, format, image}` where `image` is base64 webp, not an OpenAI-style URL.
- **PERF-4:** Centralised HTTP timeout/keepalive defaults in
  `const.py` so they can be tuned without touching `client.py`.

### Removed
- **#46:** `size`, `quality`, and `style` are no longer accepted. Those were OpenAI image enums. Automations that passed them need a real image model id and, if that model supports it, aspect ratio, resolution, or width and height.

### Fixed
- **#33:** Add the missing `config_entry_not_loaded` exception translation so a service call against an unloaded entry shows a translated error instead of the raw key.
- **#35:** Add `entity.sensor` translations for the diagnostic usage sensors, keyed to the existing `translation_key` values.
- **#43:** Remove hardcoded sensor `name=` so Home Assistant uses the entity translations.
- **#41:** Import `RateLimitError` directly. The `None` fallback could never run because `client.py` always defines it.
- **#38:** Treat `voluptuous-openapi` as the hard requirement it already is. Remove the unused availability guards.
- **#40:** Refuse config entries newer than version 1, and persist the 1.0 to 1.1 bump with `async_update_entry`.
- **#37:** Remove the unused in-memory conversation cache. Home Assistant already passes the chat log into the entity, and nothing wrote the local copy.
- **#39:** Register services when a config entry loads, and remove them when the last entry unloads.
- **#42:** Add a description on every options-form field, including stream response.
- **#36:** Push diagnostic sensor updates when API metrics change instead of polling every 30 seconds.
- **#45:** Remove the unused request-timeout option. Chat, streaming, TTS, STT, and image generation keep their own timeouts in `const.py`. A single slider cannot cover those waits.
- **HA 2026.10 tool calls:** every tool call failed with
  `TypeError: ToolResultContent.__init__() got an unexpected keyword
  argument 'tool_result'`, surfacing as "Unexpected error during
  conversation processing", because HA 2026.10 replaced the
  `tool_result` init field with `result: llm.ToolResult`. Tool results
  are now built by `_make_tool_result_content`, which uses `result=` on
  2026.10+ and falls back to both older signatures; chat-log conversion
  reads `result.data` instead of the deprecated `tool_result` property.
- **MED-3:** Conversation streaming flag (`CONF_STREAM_RESPONSE`)
  honoured by the conversation entity; opt-in default off.
- **SEC-1 (regression):** `_sanitize_header_value` in `client.py`
  previously called `.strip()` and filtered every character with
  `ord(ch) >= 0x20` on the API key. That combination silently mutated
  previously-valid keys (e.g. by trimming surrounding whitespace or
  NBSP introduced during copy-paste) into byte-different strings, which
  Venice rejected with HTTP 401 on every request — observed as
  `Venice AI HTTP error 401:` in `client.py` and
  `Invalid API key (streaming chat)` in `conversation.py`. The helper
  now removes only `\r` and `\n` (the actual header-injection vectors)
  and the client stores the unmodified `api_key` on `self._api_key`,
  applying the scrub at header-construction time only. Users who
  re-entered their key during the affected window do not need to do so
  again. Reported against commit `64b115c`.

## [0.9.0] — Initial public release

### Added
- Conversation agent backed by Venice AI Chat Completions.
- AI Task entity (`ai_task.py`) using
  `genai.process_image`/`process_text`.
- TTS entity (`tts.py`) using Venice Speech API.
- STT entity (`stt.py`) using Venice Transcriptions API.
- Coordinator-driven sensor reporting token usage and last-error.
- Reauthentication and reconfiguration flows.
- Diagnostics export with redacted API key.
- 25 unit tests covering `client.py` and `venice_api.py`.

[Unreleased]: https://github.com/grasponcrypto/venice_ai/compare/1.1.0-dev.3...HEAD
[1.1.0-dev.3]: https://github.com/grasponcrypto/venice_ai/compare/1.1.0-dev.2...1.1.0-dev.3
[1.1.0-dev.2]: https://github.com/grasponcrypto/venice_ai/compare/1.1.0-dev.1...1.1.0-dev.2
[1.1.0-dev.1]: https://github.com/grasponcrypto/venice_ai/compare/1.1.0-dev...1.1.0-dev.1
[1.1.0-dev]: https://github.com/grasponcrypto/venice_ai/compare/1.0.0...1.1.0-dev
[0.9.0]: https://github.com/grasponcrypto/venice_ai/releases/tag/0.9.0
