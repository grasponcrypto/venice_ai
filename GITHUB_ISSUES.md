# GitHub Issues — Venice AI Integration
# Prioritized by UX impact. Copy-paste each section into a new GitHub issue.
# Repo: grasponcrypto/venice_ai

---

## Issue 1 of 17
**Title:** M1: Replace custom httpx client with official openai Python SDK
**Labels:** `enhancement`

**Body:**

## Problem

Venice AI exposes an OpenAI-compatible API. The HA `openai_conversation` integration delegates all HTTP transport to the official `openai` Python SDK (`AsyncOpenAI`), which handles SSE streaming, retry semantics, schema validation, and SDK-level compatibility updates automatically. The Venice AI integration reimplements this from scratch with a custom `httpx`-based client (~800 LOC in `client.py`).

This creates several downstream risks:
- Any subtle Venice AI API deviation from the OpenAI spec must be handled manually
- The SSE parser in `client.py` (`aiter_lines`) has no reconnect logic; the official SDK handles partial-chunk boundary issues
- As Venice AI evolves its API, the custom client must be updated separately
- Adds ~800 LOC of maintenance burden

## Reference

`openai_conversation` instantiates `openai.AsyncOpenAI(api_key=..., http_client=get_async_client(hass))`. Both SDK and HA share one httpx session.

## Recommendation

Add `openai>=1.40.0` to `requirements` in `manifest.json` and replace `AsyncVeniceAIClient` with `openai.AsyncOpenAI(base_url="https://api.venice.ai/api/v1", api_key=..., http_client=get_async_client(hass))`. Streaming, retry, and SSE are all handled by the SDK.

## Files Affected

- `custom_components/venice_ai/client.py` — replace or significantly simplify
- `custom_components/venice_ai/venice_api.py` — update to use SDK types
- `custom_components/venice_ai/manifest.json` — add openai requirement
- `custom_components/venice_ai/conversation.py` — update service calls
- `custom_components/venice_ai/tts.py` — update TTS calls
- `custom_components/venice_ai/stt.py` — update STT calls (may remain custom if Venice STT is not OpenAI-compatible)

---

## Issue 2 of 17
**Title:** H1: Missing `config_entry_not_loaded` translation key causes runtime error
**Labels:** `bug`

**Body:**

## Problem

Two service handlers (`render_image`, `generate_data`) in `__init__.py` (lines ~134-137, ~180-184) raise `ServiceValidationError` with `translation_key="config_entry_not_loaded"`, but this key does **not exist** in `strings.json` or `translations/en.json`. When `entry.runtime_data` is `None`, HA will fail to look up the translation and may surface a confusing or empty error to the user.

## Reference

All reference integrations define every translation key they reference before shipping.

## Fix

Add to `strings.json` under `"exceptions"`:

```json
"config_entry_not_loaded": {
  "message": "The Venice AI config entry {config_entry} is not loaded."
}
```

Also add the same to `translations/en.json`.

---

## Issue 3 of 17
**Title:** L5: Missing `reauth_confirm` step strings causes blank reauth UI
**Labels:** `bug`, `ux`

**Body:**

## Problem

The `abort.reauth_successful` key is present in translation files, but there is no `config.step.reauth` or `config.step.reauth_confirm` entry. When HA triggers the reauth flow (which it does on `AuthenticationError` via `entry.async_start_reauth(hass)` in `__init__.py`), the flow step UI will have no title or description — presenting a blank dialog to the user.

## Fix

Add a `reauth_confirm` step to the `config` section of `strings.json` and `translations/en.json`:

```json
"reauth_confirm": {
  "title": "Re-authenticate Venice AI",
  "description": "Please enter your Venice AI API key to reconnect.",
  "data": {
    "api_key": "Venice AI API Key"
  }
}
```

---

## Issue 4 of 17
**Title:** L1: Missing `entity.sensor` translations breaks sensor i18n
**Labels:** `bug`, `ux`

**Body:**

## Problem

The six diagnostic sensors have `translation_key` values (e.g. `"request_count"`, `"error_count"`, `"total_tokens"`) set in `sensor.py`, but the translation files have no `entity.sensor.*` section. HA uses these keys to display localized sensor names in the UI. Without them, sensor names fall back to the hardcoded `name` attribute (e.g. `"API requests"`), losing i18n support.

## Fix

Add to `strings.json` and `translations/en.json`:

```json
"entity": {
  "sensor": {
    "request_count": { "name": "API Requests" },
    "error_count": { "name": "API Errors" },
    "total_tokens": { "name": "Total Tokens" },
    "prompt_tokens": { "name": "Prompt Tokens" },
    "completion_tokens": { "name": "Completion Tokens" },
    "last_error": { "name": "Last Error" }
  }
}
```

---

## Issue 5 of 17
**Title:** H2: Diagnostic sensors use polling instead of push-based updates
**Labels:** `enhancement`, `performance`

**Body:**

## Problem

The six diagnostic sensors in `sensor.py` (line 117) are polling-based (`_attr_should_poll = True`). HA calls `async_update` every 30 seconds for each sensor even though the underlying `VeniceAIMetrics` object is updated synchronously in-process with every API call. This wastes HA's scheduler resources and introduces a 30-second lag between an API call and the sensor reflecting it.

## Reference

The anthropic and openai_conversation integrations do not have diagnostic sensors, but when HA integrations track in-process counters they use `async_write_ha_state()` to push updates immediately, or the entity listens to the coordinator (`CoordinatorEntity`).

## Fix

Either:
- (a) Subclass `CoordinatorEntity` and update on coordinator refresh, or
- (b) Have `VeniceAIMetrics.record_request/record_error/record_usage` call a registered callback that triggers `async_write_ha_state()`.

Option (b) is preferable for real-time accuracy. The simplest minimal change is to set `_attr_should_poll = False` and have the entity register a metrics-change callback.

---

## Issue 6 of 17
**Title:** M2: Remove dead in-memory chat history; defer to HA ChatLog
**Labels:** `enhancement`, `architecture`

**Body:**

## Problem

The entity maintains conversation history in an `OrderedDict` (`_chat_histories`) keyed by `conversation_id` in `conversation.py`. This is lost when HA restarts or the integration reloads. Users lose context mid-conversation across restarts.

## Reference

HA's `conversation` component's `ChatLog` is designed to persist conversation content through the `conversation.ConversationInput.conversation_id` lifecycle. The `openai_conversation` entity does **not** maintain its own history dict; it relies entirely on the `ChatLog` object passed in by HA's conversation infrastructure, which handles persistence.

## Recommendation

Remove the `_chat_histories` dict entirely. Trust the `ChatLog` provided by the `_async_handle_message` call. If history accumulation within a session is needed, append to the passed-in `ChatLog` rather than a separate dict.

Note: With the M7 fix already merged (delegating prompt assembly to `chat_log.async_provide_llm_data`), the `_chat_histories` dict may already be partially unused. Audit and remove all remaining references.

---

## Issue 7 of 17
**Title:** M3: `voluptuous-openapi` listed as hard requirement but treated as optional
**Labels:** `bug`, `cleanup`

**Body:**

## Problem

`manifest.json` declares `"requirements": ["voluptuous-openapi>=0.0.4"]`, which means HA will always install it before loading the integration. Yet the code uses `HAS_VOLUPTUOUS_OPENAPI` guards everywhere and gracefully degrades when the package is absent. The fallback code paths are dead code that will never execute (since the package is guaranteed present by the manifest). Worse, if `voluptuous-openapi` is not installed for any reason (e.g. pip conflict), HA will refuse to load the component entirely rather than falling back — contradicting the intent of the optional checks.

## Fix

Choose one approach:
- **Hard requirement (recommended):** Keep it in `requirements`, remove all `HAS_VOLUPTUOUS_OPENAPI` guards, always use `voluptuous_convert`.
- **Soft requirement:** Remove from `requirements`, keep the guards, and accept degraded LLM tool schema support when absent.

---

## Issue 8 of 17
**Title:** M4: Services registered globally in `async_setup` instead of per-entry
**Labels:** `enhancement`, `architecture`

**Body:**

## Problem

`generate_image` and `ai_task` services are registered once in `async_setup` (before any config entry loads) in `__init__.py`. This is a legacy pattern. Modern HA integrations register services in `async_setup_entry` and unregister them in `async_unload_entry` so that service availability tracks config entry state.

The current approach means the services appear even if all Venice AI config entries fail to load (e.g. bad API key). A call to `generate_image` will reach the handler but fail because `entry.runtime_data` is None — which the code handles, but the service should not be registered at all in that case.

## Reference

The openai_conversation `async_setup` only does minimal work; platform-specific services are in setup_entry.

## Recommendation

Move service registration inside `async_setup_entry` with a guard to register only once (using `hass.data` as a flag), and unregister on last `async_unload_entry`.

---

## Issue 9 of 17
**Title:** M5: `async_migrate_entry` version guard logic is incorrect
**Labels:** `bug`

**Body:**

## Problem

The migration guard in `__init__.py` (lines 548-551) is:

```python
if entry.version > 1 or (entry.version == 1 and entry.minor_version >= 1):
    return True
```

This means the function returns `True` immediately for any entry already at version 1.1 or higher — it never runs migration steps below. But for entries coming in at version 1.0, `minor_version` might be `0`, which falls through to set `entry.version = 1; entry.minor_version = 1`. However, HA calls `async_migrate_entry` only when the entry version does NOT match the current version — so the guard `entry.version > 1` is incorrect for forward compatibility (if the code is downgraded, it will try to migrate a newer entry).

## Reference

The canonical HA migration pattern uses `if entry.version == 1: ... hass.config_entries.async_update_entry(entry, version=2)` with explicit step-by-step bumps.

## Fix

Rewrite the migration guard to follow the canonical pattern with explicit version checks and `hass.config_entries.async_update_entry()` calls.

---

## Issue 10 of 17
**Title:** M6: Dead code — `RateLimitError = None` import fallback
**Labels:** `cleanup`

**Body:**

## Problem

In `__init__.py` (lines 46-49):

```python
try:
    from .client import RateLimitError
except ImportError:
    RateLimitError = None  # type: ignore
```

`RateLimitError` is always defined in `client.py`. This fallback can never be triggered unless `client.py` is replaced with a stripped version. The downstream `isinstance(cause, RateLimitError)` check is guarded with `if RateLimitError is not None`, adding unnecessary complexity. This is dead code.

## Fix

Remove the try/except; import `RateLimitError` directly with the other client imports. Remove the `if RateLimitError is not None` guard.

---

## Issue 11 of 17
**Title:** L2: Missing `data_description` fields in options flow strings
**Labels:** `enhancement`, `ux`

**Body:**

## Problem

HA 2024.x+ supports `data_description` entries alongside `data` in flow steps to provide per-field helper text in the UI. All reference integrations (openai_conversation, anthropic) use these to explain what `temperature`, `top_p`, `max_tokens`, etc. mean. The Venice AI options flow has 14 fields with no descriptions, making the UI opaque to non-technical users.

## Recommendation

Add a `data_description` sibling to `data` under `options.step.init` in `strings.json` and `translations/en.json` with brief descriptions for each field, especially `strip_thinking_response`, `disable_thinking`, `max_tool_iterations`, and `stt_timestamps`.

---

## Issue 12 of 17
**Title:** L3: Missing `quality_scale` in `manifest.json`
**Labels:** `enhancement`

**Body:**

## Problem

HA's official integrations declare their `quality_scale` (e.g. `"silver"`, `"gold"`, `"platinum"`). While this is not required for HACS integrations, it signals to users and CI pipelines what level of compliance to expect and which quality checks to run.

## Recommendation

For a HACS integration of this complexity, `"silver"` is appropriate. Add to `manifest.json`:

```json
"quality_scale": "silver"
```

---

## Issue 13 of 17
**Title:** L4: Redundant `name=` alongside `translation_key=` in sensor definitions
**Labels:** `cleanup`

**Body:**

## Problem

Each `VeniceAISensorDescription` in `sensor.py` (lines 43-96) sets both `name=` (e.g. `"API requests"`) and `translation_key=` (e.g. `"request_count"`). When `translation_key` is set, HA ignores `name` and uses the translation file. Since the translation file currently lacks the `entity.sensor` section (see L1), HA silently uses the `name` field as a fallback — masking the fact that translations are missing. Once L1 is fixed, the redundant `name=` fields can be removed.

## Fix

After L1 is resolved, remove the `name=` parameter from all sensor descriptions.

---

## Issue 14 of 17
**Title:** L6: `MAX_CHAT_LOG_LENGTH` naming and purpose ambiguous
**Labels:** `cleanup`

**Body:**

## Problem

`MAX_CHAT_LOG_LENGTH = 50` is defined in `const.py` (line 62). With the adoption of HA's `ChatLog` pattern, the integration may not need to manually truncate the chat log. If the `OrderedDict` in-memory approach (M2) is refactored away, this constant becomes dead code. Currently it's used to trim the `messages` list sent to the API, which is legitimate, but the constant name is misleading — it limits the API payload, not the HA chat log.

## Recommendation

Rename to `MAX_API_MESSAGES` for clarity, or remove if M2 is implemented.

---

## Issue 15 of 17
**Title:** L7: Two timeout systems; user-configurable timeout not wired to client
**Labels:** `enhancement`

**Body:**

## Problem

`CONF_REQUEST_TIMEOUT` and `RECOMMENDED_REQUEST_TIMEOUT = 60.0` are defined in `const.py` and imported in `config_flow.py`, but `DEFAULT_HTTP_TIMEOUT = 30.0` (the base httpx client timeout) and the per-operation timeouts (`DEFAULT_CHAT_TIMEOUT`, `DEFAULT_CHAT_STREAM_TIMEOUT`, etc.) are baked into `client.py` constants and cannot be user-configured. There are two separate timeout systems that may conflict.

## Recommendation

Unify the timeout system. Either wire `CONF_REQUEST_TIMEOUT` through to the client, or remove the user-facing config option and document the hardcoded values.

---

## Issue 16 of 17
**Title:** L9: Image generation service hardcodes model to `"default"`
**Labels:** `enhancement`

**Body:**

## Problem

The `render_image` service in `__init__.py` (line 143) always calls `client.images.generate(model="default", ...)`. Users cannot select an image generation model. Venice AI offers multiple image models.

## Recommendation

The service schema should expose a `model` selector populated from the coordinator's image model list (as it does for text/audio models in the options flow).

---

## Issue 17 of 17
**Title:** L11: `unique_id` not explicitly set on `ConversationEntity`
**Labels:** `bug`

**Body:**

## Problem

The `VeniceAIConversationEntity` in `conversation.py` does not explicitly set `_attr_unique_id`. The `ConversationEntity` base class in HA may derive it from the config entry, but the `openai_conversation` entity explicitly sets:

```python
self._attr_unique_id = entry.entry_id
```

Without an explicit `unique_id`, entity registry persistence may behave unexpectedly across reloads.

## Fix

Add `self._attr_unique_id = entry.entry_id` to the entity's `__init__` method.

---

## Issue 18 of 19
**Title:** UX1: Model lacks entity awareness — `HassListEntities` tool missing from Assist API
**Labels:** `bug`, `ux`

**Body:**

## Problem

Debug logs show only 5 tools being sent to the API:
```
Tools being sent to API: ['HassTurnOn', 'HassTurnOff', 'HassCancelAllTimers', 'HassBroadcast', 'GetDateTime']
```

Notably missing is `HassListEntities`, which would allow the model to dynamically discover available entities. Without this tool AND without entity states properly injected into the system prompt, the model is effectively blind — it cannot answer questions like "what is the temperature upstairs?" because it has no way to discover or query entities.

## Expected Behavior

The Assist API should expose `HassListEntities` so models can discover entities dynamically. This is especially important for smaller/less-tuned models that may not understand the entity state format in the system prompt.

## Debug Evidence

```
2026-06-25 00:29:29.661 DEBUG Tools being sent to API: ['HassTurnOn', 'HassTurnOff', 'HassCancelAllTimers', 'HassBroadcast', 'GetDateTime']
```

Model response confirms it can't see entities:
> "I do not have a direct visual interface to 'see' a list of entities, but I can interact with any entity you provide."

## Investigation Needed

1. Check if `HassListEntities` is being filtered out somewhere in the integration
2. Verify the `llm.async_get_api()` call is requesting the correct tools
3. Check if this is a voice assistant configuration issue (user may need to expose more tools)

---

## Issue 19 of 19
**Title:** UX2: Smaller models fail at HA tool-calling — consider model guidance or fallback
**Labels:** `enhancement`, `ux`

**Body:**

## Problem

Debug logs show `google-gemma-4-31b-it` (a 31B parameter model) struggling with HA's tool-calling patterns:

1. Model says "I do not have a direct visual interface to 'see' a list of entities" — failing to understand the system prompt format
2. Model attempts to call `HassGetState` which is intentionally excluded from AssistAPI tools
3. After tool call fails, model apologizes instead of trying alternative approaches

The COMPARED_ANALYSIS.md notes:
> General-purpose models (Gemma, Mistral) that aren't HA fine-tuned fall back to tool-calling because they don't recognise the prompt structure

## Recommendation

Consider one or more of:
1. **Model guidance in system prompt**: Add explicit instructions like "Entity states are provided in your context. Do not call HassGetState — read states from the context instead."
2. **Model recommendation in UI**: Warn users when selecting smaller models that they may not work well with HA's tool-calling patterns
3. **Fallback behavior**: When a tool call fails, provide a more helpful error that guides the model (e.g., "HassGetState is not available. Entity states are in your system context — read from there.")

Note: Issue #10 (M6) already addresses improving the "tool not found" error message, which partially mitigates this.

## Debug Evidence

```
Model: google-gemma-4-31b-it
Response: "I do not have a direct visual interface to 'see' a list of entities..."
Tool call attempted: HassGetState (not in available tools)
```

---

## Summary

| # | ID | Severity | Title |
|---|---|---|---|
| 1 | M1 | 🟠 MEDIUM | Replace custom httpx client with official openai Python SDK |
| 2 | H1 | 🔴 HIGH | Missing `config_entry_not_loaded` translation key |
| 3 | L5 | 🟡 LOW | Missing `reauth_confirm` step strings |
| 4 | L1 | 🟡 LOW | Missing `entity.sensor` translations |
| 5 | H2 | 🔴 HIGH | Diagnostic sensors use polling instead of push |
| 6 | M2 | 🟠 MEDIUM | Remove dead in-memory chat history |
| 7 | M3 | 🟠 MEDIUM | `voluptuous-openapi` hard req but treated as optional |
| 8 | M4 | 🟠 MEDIUM | Services registered globally instead of per-entry |
| 9 | M5 | 🟠 MEDIUM | `async_migrate_entry` version guard logic incorrect |
| 10 | M6 | 🟠 MEDIUM | Dead `RateLimitError = None` import fallback |
| 11 | L2 | 🟡 LOW | Missing `data_description` in options flow |
| 12 | L3 | 🟡 LOW | Missing `quality_scale` in manifest |
| 13 | L4 | 🟡 LOW | Redundant `name=` alongside `translation_key=` |
| 14 | L6 | 🟡 LOW | `MAX_CHAT_LOG_LENGTH` naming ambiguous |
| 15 | L7 | 🟡 LOW | Two timeout systems conflict |
| 16 | L9 | 🟡 LOW | Image model hardcoded to `"default"` |
| 17 | L11 | 🟡 LOW | `unique_id` not set on ConversationEntity |
| 18 | UX1 | 🔴 HIGH | `HassListEntities` tool missing from Assist API |
| 19 | UX2 | 🟠 MEDIUM | Smaller models fail at HA tool-calling |

**Already completed:** M7 (delegate prompt assembly to HA `async_provide_llm_data`) — merged in commit `c6e7a85`.
