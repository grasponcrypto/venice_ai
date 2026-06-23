# Changelog

All notable changes to **Venice AI Conversation** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

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
- **PERF-4:** Centralised HTTP timeout/keepalive defaults in
  `const.py` so they can be tuned without touching `client.py`.

### Fixed
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

[Unreleased]: https://github.com/grasponcrypto/venice_ai/compare/0.9.0...HEAD
[0.9.0]: https://github.com/grasponcrypto/venice_ai/releases/tag/0.9.0
