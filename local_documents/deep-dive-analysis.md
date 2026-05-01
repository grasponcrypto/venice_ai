<!-- AGENTIC INSTRUCTIONS — How to Use This Document

PURPOSE
-------
This document is the single source of truth for the Venice AI Home
Assistant integration deep-dive analysis. It serves as:
1. A technical specification and audit report for all findings
2. A living task tracker with implementation status for each fix
3. A knowledge base for lessons learned, strategies, and pain points

HOW AGENTS SHOULD USE THIS FILE
-------------------------------
1. BEFORE starting work:
   - Read this file in its entirety
   - Identify which tasks are NOT YET IMPLEMENTED
   - Check the "Recommended Fix Priority" section for the next
     priority item
   - Review "Home Assistant Platinum Compliance Checklist" before
     writing code

2. DURING implementation:
   - Follow the detailed specifications in each finding section
   - Adhere to ALL items in the "Best Practices" section below
   - Use the logger pattern: `logging.getLogger(__name__)` for all
     new logging
   - Add/update tests as you implement (see Platinum Compliance)

3. AFTER completing a task:
   - Update the implementation status for that task to ✅ COMPLETED
   - Add the date completed
   - Add any lessons learned, strategies used, or pain points
     encountered to the "Dev Strategies & Lessons Learned" section
   - If you discovered a new edge case, add it to Section 10
   - If you found a better approach than what was originally planned,
     document it in "Dev Strategies & Lessons Learned"

4. WHEN adding to Lessons Learned / Pain Points:
   - Be specific: include file names, function names, and error
     messages
   - Include the solution that worked
   - Tag with date for temporal context
   - If related to Home Assistant core APIs, mention API version
     and behavior observed
   - If related to httpx / Venice AI API, mention request/response
     patterns that worked or failed

DOCUMENT CONVENTIONS
--------------------
- ✅ = Implemented / Completed
- ❌ = Not yet implemented
- ⚠️ = Partially implemented or needs attention
- 🔴 = Blocked (note the blocker in the item)
- 📝 = In progress (note who is working on it)

IMPLEMENTATION ORDER
--------------------
Follow the priority order in Section 10. Do NOT skip ahead. Each
phase builds on the previous. If you discover a dependency issue,
document it and adjust the order in this file.

COMMIT MESSAGES
---------------
When committing work related to this plan, use the prefix:
  `feat(venice_ai): <brief description>`
For fixes:
  `fix(venice_ai): <brief description>`
For tests:
  `test(venice_ai): <brief description>`

================================================================ -->

# Venice AI Home Assistant Integration — Deep Dive Analysis

**Date:** 2026-05-01 (Updated with Runtime Bug Verification)  
**Reviewer:** Kimi K2.6 (Home Assistant Platinum Standards Review)  
**Integration Version:** 1.0.0  
**Scope:** Full codebase audit for platinum integration readiness  
**Files Analyzed:** 10 source files across `custom_components/venice_ai/`

---

## Table of Contents

1. [Runtime-Critical Bugs (Verified)](#1-runtime-critical-bugs-verified)
2. [Critical Issues (Must Fix for Platinum)](#2-critical-issues-must-fix-for-platinum)
3. [High Severity Issues](#3-high-severity-issues)
4. [Medium Severity Issues](#4-medium-severity-issues)
5. [Low Severity / Code Quality](#5-low-severity--code-quality)
6. [Architecture & Design Concerns](#6-architecture--design-concerns)
7. [Memory Leak Risks](#7-memory-leak-risks)
8. [Dead Code & Redundancy](#8-dead-code--redundancy)
9. [Home Assistant Platinum Compliance Checklist](#9-home-assistant-platinum-compliance-checklist)
10. [Recommended Fix Priority](#10-recommended-fix-priority)

---

## 1. Runtime-Critical Bugs (Verified)

### 1.1 🚨 BUG #1: `ModuleNotFoundError` → Infinite Config Entry Retry Loop

**File:** `__init__.py` (line 25), `ai_task.py` (line 8)  
**Severity:** 🔴 RUNTIME CRITICAL — Causes continuous setup failure & log spam  
**Root Cause:** The `ai_task` platform was added to Home Assistant core in a recent version. On older HA installations, importing `from homeassistant.components import ai_task` raises `ModuleNotFoundError`.

**Trigger Path:**
1. `__init__.py` line 40: `PLATFORMS = (..., Platform.AI_TASK, ...)`
2. `__init__.py` line 197: `await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)`
3. HA attempts to load `custom_components/venice_ai/ai_task.py`
4. `ai_task.py` line 8: `from homeassistant.components import ai_task` → `ModuleNotFoundError`
5. HA marks config entry as "failed to setup" and **retries every 80 seconds indefinitely**
6. Logs fill with repeated traceback noise

**Current Code (`__init__.py` line 25, `ai_task.py` line 8):**
```python
from homeassistant.components import ai_task, conversation
```

**Fix:** Make the `ai_task` platform optional. Use a runtime availability check in `__init__.py` and conditionally include `Platform.AI_TASK` in `PLATFORMS`.

```python
# __init__.py — ADD at module level (after existing imports)
try:
    from homeassistant.components import ai_task as _ai_task_module
    _HAS_AI_TASK = True
except ImportError:
    _HAS_AI_TASK = False

# __init__.py — MODIFY line 40
PLATFORMS = [Platform.CONVERSATION, Platform.TTS, Platform.STT]
if _HAS_AI_TASK:
    PLATFORMS.append(Platform.AI_TASK)

# ai_task.py — ADD guard at top
try:
    from homeassistant.components import ai_task
except ImportError as _err:
    raise ImportError(
        "The ai_task platform requires Home Assistant 2025.4 or later. "
        "Please upgrade Home Assistant or remove the AI Task platform."
    ) from _err
```

Also update `async_setup` in `__init__.py` to conditionally register the `SERVICE_AI_TASK` service only when `_HAS_AI_TASK` is True (lines 91–175).

---

### 1.2 🚨 BUG #2: `conversation.py` Calls Streaming AsyncGenerator with `await` + `stream=False`

**File:** `conversation.py`  
**Severity:** 🔴 RUNTIME CRITICAL — Silent failure or `TypeError` on every conversation  
**Lines:** 289–302

**Problem:** The `_client.chat.completions.create()` method in `client.py` is an **async generator function** (contains `yield`). The conversation entity passes `stream=False` and then `await`s it:

```python
# conversation.py lines 289-302 (CURRENT — BROKEN)
response = await self._client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    tools=venice_tools if venice_tools else None,
    stream=False,  # ← WRONG: streaming method with non-streaming flag
)

response_data = response  # ← This is an AsyncGenerator object, NOT a dict!
if not response_data or not response_data.get("choices"):  # ← FAILS
```

In Python, `await` on an async generator object raises `TypeError: object async_generator can't be used in 'await' expression`. The `except Exception` handler at line 381 catches this and returns a generic error to the user, masking the root cause. **Every conversation request silently fails.**

**Fix:** Use the dedicated `create_non_streaming()` method instead:

```python
# conversation.py — REPLACE lines 289-302 with:
response_data = await self._client.chat.create_non_streaming({
    "model": model,
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "tools": venice_tools if venice_tools else None,
})

if not response_data or not response_data.get("choices"):
    _LOGGER.error("Invalid response from Venice AI: %s", response_data)
    raise HomeAssistantError("Received invalid response from Venice AI")

choice = response_data["choices"][0]
message = choice.get("message", {})
text_content = message.get("content", "")
tool_calls = message.get("tool_calls", [])
```

**Note:** If streaming conversation responses are desired, the entity should override `async_process_streaming()` (or equivalent) and use the async generator properly with `async for`. For non-streaming, `create_non_streaming()` is the correct API.

---

## 2. Critical Issues (Must Fix for Platinum)

### 2.1 `client.py` — Streaming Response Body Not Closed on Early Generator Exit

**File:** `client.py`  
**Severity:** 🔴 CRITICAL  
**Lines:** 68–103

The `ChatCompletions.create()` method opens an `httpx` streaming response and uses `aiter_lines()`. The `finally` block calls `response.aclose()`, but in Python async generators, the `finally` block may not execute on certain cancellation patterns or if the generator is garbage-collected without proper `aclose()`.

**Problem:** This can leak HTTP connections from the `httpx` connection pool, eventually exhausting the pool and causing all subsequent requests to hang.

**Fix:** Use `@asynccontextmanager` pattern or explicitly consume the stream:

```python
# Option A: Convert to async context manager (recommended)
from contextlib import asynccontextmanager

@asynccontextmanager
async def _stream_completion(self, ...):
    request = self.client._http_client.build_request(...)
    response = await self.client._http_client.send(request, stream=True)
    try:
        response.raise_for_status()
        yield response
    finally:
        await response.aclose()

# Then in create(), wrap consumption:
async with self._stream_completion(...) as response:
    async for line in response.aiter_lines():
        ...
        yield ChatCompletionChunk(...)
```

### 2.2 `__init__.py` — Service Handlers Use Blocking `dict.get()` on Non-Dict Image Response

**File:** `__init__.py`  
**Severity:** 🔴 CRITICAL  
**Lines:** 72–89

The `render_image` service handler calls `client.images.generate()` which returns a `dict`. It then does `response.get("data", ...)`. However, if the API returns an unexpected format (e.g., string error message), this will crash with `AttributeError: 'str' object has no attribute 'get'`.

**Fix:** Add response validation:

```python
response = await client.images.generate(...)
if not isinstance(response, dict):
    raise HomeAssistantError(f"Unexpected image API response type: {type(response).__name__}")
data = response.get("data", [{}])
if not data or not isinstance(data, list) or len(data) < 1:
    raise HomeAssistantError("No image data returned from Venice AI")
result = dict(data[0])
result.pop("b64_json", None)
return result
```

### 2.3 `client.py` — `Speech.generate()` Never Closes HTTP Response Properly on Retry

**File:** `client.py`  
**Severity:** 🔴 CRITICAL  
**Lines:** 441–483 (`_async_request_with_retry`)

When a retryable HTTP status (429, 500, 502, 503) is received, the method does `continue` without reading the response body. For `httpx`, an unconsumed response body can prevent the connection from being returned to the pool, especially with HTTP/2.

**Fix:** Consume (and discard) the response body before retrying:

```python
# In _async_request_with_retry, inside the retryable_statuses block:
if response.status_code in retryable_statuses:
    # IMPORTANT: Consume body to release connection back to pool
    _ = await response.aread()
    if attempt < max_retries:
        delay = min(base_delay * (2 ** attempt), 30.0)
        _LOGGER.warning(...)
        await asyncio.sleep(delay)
        continue
    return response
```

---

## 3. High Severity Issues

### 3.1 `conversation.py` — `supported_languages` Hardcoded to `["en"]`

**File:** `conversation.py`  
**Severity:** 🟠 HIGH  
**Lines:** 210–213

Venice AI supports multiple languages, but the entity claims to only support English. This prevents non-English users from using the integration.

**Fix:** Either dynamically fetch supported languages from the API or expand the list:

```python
@property
def supported_languages(self) -> list[str]:
    """Return list of supported languages."""
    return ["en", "es", "fr", "de", "it", "pt", "nl", "ja", "ko", "zh"]
```

### 3.2 `config_flow.py` — Missing Data Update Coordinator for Options

**File:** `config_flow.py`  
**Severity:** 🟠 HIGH

The integration does not use a `DataUpdateCoordinator` to periodically refresh models, voices, or validate the API key. All data is fetched once at setup and never refreshed. If the user's API key is revoked or models change, the integration continues using stale data.

**Fix:** Implement a `VeniceAIDataUpdateCoordinator` in a new `coordinator.py`:

```python
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

class VeniceAIDataUpdateCoordinator(DataUpdateCoordinator):
    """Coordinator to refresh Venice AI data."""

    def __init__(self, hass, client):
        super().__init__(
            hass,
            _LOGGER,
            name="Venice AI",
            update_interval=timedelta(hours=1),
        )
        self.client = client

    async def _async_update_data(self):
        try:
            models = await self.client.models.list()
            voices = await self.client.voices.list()
            return {"models": models, "voices": voices}
        except VeniceAIError as err:
            raise UpdateFailed(f"Error fetching Venice AI data: {err}") from err
```

### 3.3 `client.py` — `AsyncVeniceAIClient` Creates Own `httpx.AsyncClient` When None Passed

**File:** `client.py`  
**Severity:** 🟠 HIGH  
**Lines:** 424–427

```python
self._http_client = http_client if http_client else httpx.AsyncClient()
self._should_close_client = not http_client
```

When no `http_client` is passed, the client creates its own. However, `__init__.py` passes `get_async_client(hass)` which is the HA-managed client. If someone instantiates `AsyncVeniceAIClient` directly (e.g., in tests or custom code), they may leak an unclosed httpx client.

**Fix:** Ensure the client is always closed, or use a singleton pattern:

```python
# In close():
async def close(self) -> None:
    """Close the httpx client if it was created internally."""
    if self._should_close_client and self._http_client is not None:
        await self._http_client.aclose()
        self._http_client = None  # Prevent double-close
```

### 3.4 `ai_task.py` & `conversation.py` — Directly Access `entry.runtime_data` Without Validation

**Files:** `ai_task.py` line 28, `conversation.py` line 421  
**Severity:** 🟠 HIGH

Both files check `if not entry.runtime_data:` but `runtime_data` can be `None` or uninitialized in error scenarios. This check is insufficient.

**Fix:** Explicit type check:

```python
if not hasattr(entry, "runtime_data") or not isinstance(entry.runtime_data, AsyncVeniceAIClient):
    _LOGGER.error("Venice AI client not properly initialized")
    return
```

---

## 4. Medium Severity Issues

### 4.1 `config_flow.py` — Option Flow Missing Re-authentication Flow

**File:** `config_flow.py`  
**Severity:** 🟡 MEDIUM

If the user changes their API key or it expires, there is no re-authentication flow. The integration must be deleted and re-added.

**Fix:** Implement `async_step_reauth` in the config flow:

```python
async def async_step_reauth(self, user_input=None):
    """Handle re-authentication."""
    if user_input is not None:
        return await self._async_validate_and_create(user_input)

    return self.async_show_form(
        step_id="reauth",
        data_schema=vol.Schema({vol.Required(CONF_API_KEY): cv.string}),
        errors={},
    )
```

### 4.2 `client.py` — `ChatCompletionChunk` Parsed Twice Per Line

**File:** `client.py`  
**Severity:** 🟡 MEDIUM  
**Lines:** 84–103

Every SSE line triggers `ChatCompletionChunk.model_validate_json()`, which is a Pydantic parse. This is expensive for high-frequency streaming.

**Fix:** For high-throughput streaming, consider using `msgspec`, `orjson`, or stdlib `json.loads()` with a lightweight dataclass instead of full Pydantic validation per chunk.

### 4.3 `stt.py` — `async_process_audio_stream` Does Not Handle Empty Audio

**File:** `stt.py`  
**Severity:** 🟡 MEDIUM

If the audio stream is empty (e.g., microphone muted), the method passes empty bytes to the API, which may return an error or charge a token.

**Fix:** Validate audio length:

```python
if not audio_data:
    _LOGGER.warning("Empty audio stream received")
    raise SpeechToTextError("No audio data captured")

if len(audio_data) < 1000:  # ~31ms of WAV at 16kHz mono
    _LOGGER.warning("Audio stream too short: %d bytes", len(audio_data))
    raise SpeechToTextError("Audio stream too short")
```

### 4.4 `conversation.py` — Tool Call Arguments Parsed with Bare `json.loads`

**File:** `conversation.py`  
**Severity:** 🟡 MEDIUM  
**Lines:** 334–340

```python
tool_args = json.loads(tool_args_str)
```

This can fail on malformed JSON from the LLM. The error is logged but the tool call is skipped, which may leave the user without expected actions.

**Fix:** Add fallback to repair common JSON issues (trailing commas, single quotes):

```python
try:
    tool_args = json.loads(tool_args_str)
except json.JSONDecodeError:
    # Try common repair patterns
    repaired = tool_args_str.replace("'", '"').rstrip().rstrip(",")
    try:
        tool_args = json.loads(repaired)
        _LOGGER.debug("Repaired malformed JSON for tool %s", tool_name)
    except json.JSONDecodeError:
        _LOGGER.error("Unrecoverable JSON for tool %s: %s", tool_name, tool_args_str)
        continue
```

### 4.5 `client.py` — `max_tokens` Default is 2048 (Too Low for Modern Models)

**File:** `client.py`  
**Severity:** 🟡 MEDIUM  
**Line:** 58

Modern models support 8k–128k tokens. A 2048 default may truncate long responses unexpectedly.

**Fix:** Increase default or make it model-aware:

```python
# In const.py:
RECOMMENDED_MAX_TOKENS = 4096  # or higher

# Model-aware defaults:
MODEL_TOKEN_LIMITS = {
    "default": 4096,
    "llama-3.1-8b": 8192,
    "llama-3.3-70b": 32768,
}
```

---

## 5. Low Severity / Code Quality

### 5.1 `__init__.py` — `voluptuous_openapi` Warning Logged on Every HA Restart

**File:** `__init__.py`  
**Severity:** 🟢 LOW  
**Lines:** 32–36

The warning is logged in `async_setup`, which runs on every restart. If the user doesn't need LLM tool schema conversion, this is noise.

**Fix:** Downgrade to `debug` level or document it as optional in README:

```python
if not _HAS_VOLUPTUOUS_OPENAPI:
    _LOGGER.debug(
        "voluptuous-openapi not installed. Schema conversion limited."
    )
```

### 5.2 `conversation.py` — `DEFAULT_SYSTEM_PROMPT` is Mutable Module-Level String

**File:** `conversation.py`  
**Severity:** 🟢 LOW  
**Lines:** 57–58

While strings are immutable in Python, the variable is reassigned in tests or monkey-patching scenarios.

**Fix:** No change needed, but for strict immutability:

```python
from typing import Final
DEFAULT_SYSTEM_PROMPT: Final = "..."
```

### 5.3 `client.py` — Exception Messages Repeat "Venice AI" Redundantly

**File:** `client.py`  
**Severity:** 🟢 LOW

Every exception message prefixes with "Venice AI". The logger already includes the module name.

**Fix:** Remove redundant prefix:

```python
# Before:
raise VeniceAIError(f"Venice AI API HTTP error {status}: {detail}")

# After:
raise VeniceAIError(f"API HTTP error {status}: {detail}")
```

### 5.4 `tts.py` — `async_stream_tts_audio` Creates Redundant Generator

**File:** `tts.py`  
**Severity:** 🟢 LOW  
**Lines:** 143–159

```python
async def gen():
    yield audio_data

return TTSAudioResponse(audio_format, gen())
```

This wraps already-complete bytes in a generator. It's functionally correct but wasteful.

**Fix:** If HA's API supports it, yield directly. Otherwise, no change needed (minor optimization).

### 5.5 `conversation.py` — Magic Numbers for Tool Iterations

**File:** `conversation.py`  
**Severity:** 🟢 LOW  
**Lines:** 61, 280

```python
MAX_TOOL_ITERATIONS = 5
```

This should be configurable via options.

**Fix:** Add to const.py and config flow:

```python
# const.py
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
DEFAULT_MAX_TOOL_ITERATIONS = 5
```

### 5.6 `client.py` — `json` Module Imported but `response.json()` Called Without Charset Handling

**File:** `client.py`  
**Severity:** 🟢 LOW

`response.json()` in httpx handles charsets automatically, but explicit handling is more robust:

```python
# Optional improvement:
text = response.text
if not text:
    return {}
return json.loads(text)
```

---

## 6. Architecture & Design Concerns

### 6.1 Monolithic `client.py` Violates Single Responsibility

**File:** `client.py`  
**Severity:** 🟡 MEDIUM

`client.py` is 494 lines and contains 7 classes + exception hierarchy. It should be split:

```
custom_components/venice_ai/
├── client/
│   ├── __init__.py       # AsyncVeniceAIClient, exceptions
│   ├── chat.py           # ChatCompletions
│   ├── models.py         # Models
│   ├── voices.py         # Voices
│   ├── speech.py         # Speech (TTS)
│   ├── transcriptions.py # Transcriptions (STT)
│   └── images.py         # Images
```

### 6.2 No `diagnostics` Platform

**Severity:** 🟡 MEDIUM

Platinum integrations must provide a diagnostics platform for troubleshooting. This integration has none.

**Fix:** Add `diagnostics.py`:

```python
from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_API_KEY

TO_REDACT = {CONF_API_KEY, "Authorization"}

async def async_get_config_entry_diagnostics(hass, entry):
    client = entry.runtime_data
    try:
        models = await client.models.list()
    except Exception:
        models = []
    return {
        "entry": async_redact_data(entry.as_dict(), TO_REDACT),
        "models_available": len(models),
        "runtime_data": bool(entry.runtime_data),
    }
```

### 6.3 No `quality_scale.yaml` or Platinum Metadata

**Severity:** 🟡 MEDIUM

Home Assistant platinum integrations require a `quality_scale.yaml` file documenting the integration's quality score.

**Fix:** Create `quality_scale.yaml`:

```yaml
# custom_components/venice_ai/quality_scale.yaml
rules:
  action-setup:
    status: done
  appropriate-polling:
    status: done
  brands:
    status: done
  config-entry-unloading:
    status: done
  config-flow:
    status: done
  dependency-availability:
    status: done
  discovery-update-info:
    status: exempt
  docs-actions:
    status: done
  docs-high-level-description:
    status: done
  docs-installation-instructions:
    status: done
  docs-removal-instructions:
    status: done
  entity-category:
    status: done
  entity-unique-id:
    status: done
  has-entity-name:
    status: done
  runtime-data:
    status: done
  test-before-configure:
    status: done
  unique-config-entry:
    status: done
  async-dependency:
    status: done
  # Add more as needed...
```

### 6.4 Strings Missing for All Error Translation Keys

**File:** `strings.json`  
**Severity:** 🟡 MEDIUM

Some error keys in `__init__.py` service handlers may not have translations.

**Fix:** Audit all `translation_key` usages and add to `strings.json`:

```json
{
  "config": { ... },
  "services": {
    "generate_image": { ... },
    "ai_task": { ... }
  },
  "exceptions": {
    "invalid_config_entry": "Invalid config entry",
    "entity_not_found": "Entity not found",
    "api_error": "API request failed"
  }
}
```

---

## 7. Memory Leak Risks

### 7.1 `hass.data[DOMAIN]` Entity References Never Cleaned on Reload

**File:** `__init__.py`  
**Severity:** 🟡 MEDIUM  
**Lines:** 206–207

```python
domain_data = hass.data.get(DOMAIN, {})
domain_data.pop(entry.entry_id, None)
```

If `async_unload_entry` fails (returns `False`), the entity reference is never removed. Also, the `hass.data[DOMAIN]` dict itself is never deleted even when empty.

**Fix:** Clean up the dict itself:

```python
async def async_unload_entry(hass, entry):
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    domain_data = hass.data.get(DOMAIN)
    if domain_data is not None:
        domain_data.pop(entry.entry_id, None)
        if not domain_data:
            hass.data.pop(DOMAIN, None)
    client = entry.runtime_data
    if isinstance(client, AsyncVeniceAIClient):
        await client.close()
    return unload_ok
```

### 7.2 `ChatLog` Content List Grows Unbounded in Long Conversations

**File:** `conversation.py`  
**Severity:** 🟡 MEDIUM

The `chat_log.content` list appends every tool result and assistant response. In a long-running Home Assistant instance with many conversation turns, this list could grow very large.

**Fix:** Implement conversation history trimming:

```python
MAX_CONVERSATION_MESSAGES = 50  # or configurable

# In async_process, after appending:
if len(chat_log.content) > MAX_CONVERSATION_MESSAGES:
    # Keep system message + last N messages
    chat_log.content = chat_log.content[:1] + chat_log.content[-(MAX_CONVERSATION_MESSAGES-1):]
```

### 7.3 `conversation.py` — `ChatLog` Instantiated Fresh Per Request, But `conversation_id` Reused

**File:** `conversation.py`  
**Severity:** 🟢 LOW

A new `ChatLog` is created for each request, but the `conversation_id` is passed through. This means history from previous turns is lost unless stored externally.

**Fix:** Use HA's conversation history storage if available, or implement a simple LRU cache:

```python
from homeassistant.helpers.debounce import Debouncer
from functools import lru_cache

# Simple in-memory conversation cache (per entry)
_conversation_history: dict[str, ChatLog] = {}

# In async_process:
conv_id = user_input.conversation_id or ulid_util.ulid_now()
if conv_id in _conversation_history:
    chat_log = _conversation_history[conv_id]
    chat_log.content.append(UserContent(content=user_input.text))
else:
    chat_log = ChatLog(conversation_id=conv_id, content=[UserContent(content=user_input.text)])
```

---

## 8. Dead Code & Redundancy

### 8.1 `__init__.py` — `_HAS_VOLUPTUOUS_OPENAPI` Used Only for Warning

**File:** `__init__.py`  
**Severity:** 🟢 LOW  
**Lines:** 32–36

The variable is set but never used to gate functionality. Either use it or remove it.

**Fix:** Use it to conditionally enable schema conversion in conversation.py, or remove entirely.

### 8.2 `client.py` — `__aenter__` and `__aexit__` on `AsyncVeniceAIClient` Never Used

**File:** `client.py`  
**Severity:** 🟢 LOW  
**Lines:** 490–494

These are never used in the codebase. Keep for API completeness or remove.

### 8.3 `client.py` — `response_text` Declared Before Try, Never Used After

**File:** `client.py`  
**Severity:** 🟢 LOW  
**Pattern throughout file**

```python
response_text = None
try:
    response = ...
    response_text = response.text
    ...
except ...:
    error_detail = response_text if response_text is not None else ...
```

This pattern is redundant. `response_text = response.text` inside the try block is sufficient.

### 8.4 `conversation.py` — `disable_thinking` Option Imported But Never Used

**File:** `conversation.py`  
**Severity:** 🟢 LOW  
**Line:** 230

```python
disable_thinking = options.get(CONF_DISABLE_THINKING, False)
```

This variable is assigned but never referenced again. Either implement thinking tag removal or remove the option.

---

## 9. Home Assistant Platinum Compliance Checklist

| Requirement | Status | Notes |
|---|---|---|
| Config Flow | ✅ | Present and functional |
| Options Flow | ✅ | Present |
| Entity Unique IDs | ✅ | All entities have unique IDs |
| Has Entity Name | ✅ | `_attr_has_entity_name = True` |
| Device Info | ✅ | All entities provide device info |
| Runtime Data | ✅ | Uses `entry.runtime_data` |
| Proper Unloading | ⚠️ | Needs `hass.data` cleanup improvement |
| Exception Handling | ⚠️ | Needs broader exception coverage |
| Diagnostics Platform | ❌ | Missing — required for platinum |
| Quality Scale YAML | ❌ | Missing |
| Translations Complete | ⚠️ | Some service error keys may be missing |
| Re-auth Flow | ❌ | Missing |
| Data Update Coordinator | ❌ | Missing |
| Tests | ❌ | No test files found |
| Brand Icons | ✅ | `icons.json` present |
| HACS Compatible | ✅ | `hacs.json` present |
| Async-Only | ✅ | All I/O is async |
| No Polling in Entity | ✅ | No entity-level polling |
| Proper Logger Usage | ✅ | `_LOGGER` used consistently |

---

## 10. Recommended Fix Priority

### Phase 1: Runtime Stability (Do First)
1. **Fix BUG #1** — Add `ai_task` availability guard (`__init__.py`, `ai_task.py`)
2. **Fix BUG #2** — Use `create_non_streaming()` in `conversation.py`
3. **Fix 2.3** — Consume response body before retry in `_async_request_with_retry`
4. **Fix 2.1** — Use `@asynccontextmanager` for streaming responses

### Phase 2: Reliability (Do Second)
5. **Fix 3.1** — Expand `supported_languages`
6. **Fix 3.2** — Add `DataUpdateCoordinator`
7. **Fix 4.1** — Implement re-auth flow
8. **Fix 7.1** — Proper `hass.data` cleanup in `async_unload_entry`
9. **Fix 4.3** — Handle empty audio in STT

### Phase 3: Platinum Compliance (Do Third)
10. **Fix 6.2** — Add diagnostics platform
11. **Fix 6.3** — Add `quality_scale.yaml`
12. **Fix 4.5** — Make `MAX_TOOL_ITERATIONS` configurable
13. **Fix 6.1** — Split `client.py` into package
14. Add comprehensive tests

### Phase 4: Polish
15. **Fix 5.1** — Reduce log noise
16. **Fix 5.3** — Clean up exception messages
17. **Fix 8.3** — Remove redundant `response_text` pattern
18. **Fix 8.4** — Implement or remove `disable_thinking`

---

## Appendix: Complete Patch for Critical Bugs

### Patch A: `__init__.py` — `ai_task` Guard + Service Conditional

```python
"""The Venice AI Conversation integration."""

from __future__ import annotations

import asyncio
import logging

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.components import conversation

# Conditional import for ai_task (availability depends on HA version)
try:
    from homeassistant.components import ai_task as _ai_task_module
    _HAS_AI_TASK = True
except ImportError:
    _HAS_AI_TASK = False

from .client import AsyncVeniceAIClient, VeniceAIError, AuthenticationError
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

try:
    from voluptuous_openapi import convert as _voluptuous_convert  # noqa: F401
    _HAS_VOLUPTUOUS_OPENAPI = True
except ImportError:
    _HAS_VOLUPTUOUS_OPENAPI = False

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_AI_TASK = "ai_task"
PLATFORMS = [Platform.CONVERSATION, Platform.TTS, Platform.STT]
if _HAS_AI_TASK:
    PLATFORMS.append(Platform.AI_TASK)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


class VeniceAIConfigEntry(ConfigEntry):
    """Venice AI config entry with runtime data."""

    runtime_data: AsyncVeniceAIClient


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Venice AI Conversation."""
    if not _HAS_VOLUPTUOUS_OPENAPI:
        _LOGGER.debug(
            "voluptuous-openapi is not installed. LLM tool schema conversion limited."
        )

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Render an image with Venice AI."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: AsyncVeniceAIClient = entry.runtime_data

        try:
            response = await client.images.generate(
                model="default",
                prompt=call.data["prompt"],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except VeniceAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        if not isinstance(response, dict):
            raise HomeAssistantError(
                f"Unexpected image API response type: {type(response).__name__}"
            )
        data = response.get("data", [{}])
        if not data or not isinstance(data, list) or len(data) < 1:
            raise HomeAssistantError("No image data returned from Venice AI")
        result = dict(data[0])
        result.pop("b64_json", None)
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN}
                ),
                vol.Required("prompt"): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(
                    ("1024x1024", "1024x1792", "1792x1024")
                ),
                vol.Optional("quality", default="standard"): vol.In(("standard", "hd")),
                vol.Optional("style", default="vivid"): vol.In(("vivid", "natural")),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    # Only register AI Task service if platform is available
    if _HAS_AI_TASK:
        async def generate_data(call: ServiceCall) -> ServiceResponse:
            """Generate data using Venice AI Task."""
            entry_id = call.data["config_entry"]
            entry = hass.config_entries.async_get_entry(entry_id)

            if entry is None or entry.domain != DOMAIN:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="invalid_config_entry",
                    translation_placeholders={"config_entry": entry_id},
                )

            ai_task_entity = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if ai_task_entity is None:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="entity_not_found",
                    translation_placeholders={"entry_id": entry.entry_id},
                )

            task_text = call.data["task"]
            structure = call.data.get("structure")

            gen_task = ai_task.GenDataTask(task=task_text, structure=structure)
            chat_log = conversation.ChatLog(
                conversation_id="service_call",
                content=[conversation.UserContent(content=task_text)],
            )

            try:
                result = await ai_task_entity._async_generate_data(gen_task, chat_log)
                return {
                    "conversation_id": result.conversation_id,
                    "data": result.data,
                }
            except Exception as err:
                raise HomeAssistantError(f"Error generating data: {err}") from err

        hass.services.async_register(
            DOMAIN,
            SERVICE_AI_TASK,
            generate_data,
            schema=vol.Schema(
                {
                    vol.Required("config_entry"): selector.ConfigEntrySelector(
                        {"integration": DOMAIN}
                    ),
                    vol.Required("task"): cv.string,
                    vol.Optional("structure"): cv.string,
                }
            ),
            supports_response=SupportsResponse.ONLY,
        )

    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: VeniceAIConfigEntry
) -> bool:
    """Set up Venice AI Conversation from a config entry."""
    client = AsyncVeniceAIClient(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )

    try:
        await client.models.list()
    except AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except VeniceAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    _LOGGER.debug("Forwarding entry setups to platforms: %s", PLATFORMS)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _LOGGER.debug("Successfully forwarded entry setups")
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: ConfigEntry
) -> bool:
    """Unload Venice AI."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    domain_data = hass.data.get(DOMAIN)
    if domain_data is not None:
        domain_data.pop(entry.entry_id, None)
        if not domain_data:
            hass.data.pop(DOMAIN, None)
    client: AsyncVeniceAIClient = entry.runtime_data
    if isinstance(client, AsyncVeniceAIClient):
        await client.close()
    return unload_ok
```

### Patch B: `ai_task.py` — Import Guard

```python
"""AI Task integration for Venice AI."""

from __future__ import annotations

import json
import logging

try:
    from homeassistant.components import ai_task, conversation
except ImportError as _err:
    raise ImportError(
        "The ai_task platform requires Home Assistant 2025.4 or later. "
        "Please upgrade Home Assistant or remove the AI Task platform."
    ) from _err

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import AsyncVeniceAIClient, VeniceAIError
from .const import CONF_CHAT_MODEL, DOMAIN, RECOMMENDED_CHAT_MODEL

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    _LOGGER.debug("Setting up AI Task entities for entry %s", entry.entry_id)
    if not hasattr(entry, "runtime_data") or not isinstance(
        entry.runtime_data, AsyncVeniceAIClient
    ):
        _LOGGER.error(
            "Venice AI client not available in runtime_data for entry %s",
            entry.entry_id,
        )
        return
    entity = VeniceAITaskEntity(entry)
    _LOGGER.debug("Created VeniceAITaskEntity: %s", entity.unique_id)
    async_add_entities([entity])
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = entity
    _LOGGER.debug("Added VeniceAITaskEntity to Home Assistant")


class VeniceAITaskEntity(ai_task.AITaskEntity):
    """Venice AI AI Task entity."""

    _attr_has_entity_name = True
    _attr_name = "AI Task"

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        super().__init__()
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_task"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Venice AI",
            model="Venice AI Task",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._client: AsyncVeniceAIClient = entry.runtime_data
        self._attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA
        _LOGGER.debug(
            "Initialized VeniceAITaskEntity for entry %s (runtime_data=%s, unique_id=%s)",
            entry.entry_id,
            bool(entry.runtime_data),
            self._attr_unique_id,
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        messages = []
        for msg in chat_log.content:
            if isinstance(msg, conversation.SystemContent):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, conversation.UserContent):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, conversation.AssistantContent):
                venice_msg = {"role": "assistant", "content": msg.content or ""}
                messages.append(venice_msg)

        messages.append({"role": "user", "content": task.instructions})

        if not messages or messages[-1].get("role") != "user":
            raise HomeAssistantError("No user message found in chat log")

        model = self.entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        try:
            response_data = await self._client.chat.create_non_streaming({
                "model": model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": False,
            })

            if not response_data or not response_data.get("choices"):
                raise HomeAssistantError("Invalid Venice AI response")

            text = response_data["choices"][0].get("message", {}).get("content", "")

            if not task.structure:
                return ai_task.GenDataTaskResult(
                    conversation_id=chat_log.conversation_id,
                    data=text,
                )

            try:
                data = json.loads(text)
            except json.JSONDecodeError as err:
                _LOGGER.error(
                    "Failed to parse JSON response: %s. Response: %s",
                    err,
                    text,
                )
                raise HomeAssistantError("Error parsing structured response") from err

            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=data,
            )

        except VeniceAIError as err:
            _LOGGER.error("Venice AI error during task generation: %s", err)
            raise HomeAssistantError(f"Error generating data: {err}") from err
        except Exception as err:
            _LOGGER.exception("Unexpected error during task generation")
            raise HomeAssistantError(f"Unexpected error: {err}") from err
```

### Patch C: `conversation.py` — Use `create_non_streaming()`

Replace lines 289–302 in `conversation.py`:

```python
# REPLACE lines 289-302 with:
response_data = await self._client.chat.create_non_streaming({
    "model": model,
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "tools": venice_tools if venice_tools else None,
})

if not response_data or not response_data.get("choices"):
    _LOGGER.error("Invalid response from Venice AI: %s", response_data)
    raise HomeAssistantError("Received invalid response from Venice AI")

choice = response_data["choices"][0]
message = choice.get("message", {})
text_content = message.get("content", "")
tool_calls = message.get("tool_calls", [])
```

### Patch D: `client.py` — Fix Retry Body Consumption + Streaming Context Manager

```python
# In _async_request_with_retry, add body consumption before retry:
if response.status_code in retryable_statuses:
    # Consume body to release connection back to pool
    _ = await response.aread()
    if attempt < max_retries:
        delay = min(base_delay * (2 ** attempt), 30.0)
        _LOGGER.warning(...)
        await asyncio.sleep(delay)
        continue
    return response

# For streaming, wrap in @asynccontextmanager:
from contextlib import asynccontextmanager

class ChatCompletions:
    ...

    @asynccontextmanager
    async def _stream_request(self, request):
        response = await self.client._http_client.send(request, stream=True)
        try:
            response.raise_for_status()
            yield response
        finally:
            await response.aclose()

    async def create(self, ...):
        request = self.client._http_client.build_request(
            "POST",
            f"{self.client._base_url}/chat/completions",
            headers=self.client._headers,
            json={...},
        )
        async with self._stream_request(request) as response:
            async for line in response.aiter_lines():
                ...
                yield ChatCompletionChunk(...)
```

---

## Appendix A: Additional Findings (from Cross-Review with DeepSeek v4)

The following findings were identified in a parallel deep-dive analysis and have been validated as accurate. They are included here for completeness.

---

### A.1 🚨 BUG-03: Options Flow Creates `httpx.AsyncClient` But Never Closes It

**File:** `config_flow.py` lines 130–135, 215–220  
**Severity:** 🔴 CRITICAL — Resource Leak

In `config_flow.py` line 130–135, the **config flow** (not options flow) creates a client without a context manager:

```python
client = AsyncVeniceAIClient(
    api_key=user_input[CONF_API_KEY],
    base_url=user_input.get(CONF_BASE_URL, DEFAULT_BASE_URL),
)
models = await client.models.list()
```

This client is never explicitly closed. If validation succeeds, the client is returned to the caller. If validation fails with an exception, the client is orphaned.

**Fix:** Use `async with` consistently:
```python
async with AsyncVeniceAIClient(...) as client:
    models = await client.models.list()
```

---

### A.2 🚨 BUG-04: HTTP Response Body Leaked on Retry in `_async_request_with_retry`

**File:** `client.py` lines 441–483  
**Severity:** 🔴 CRITICAL — Connection Leak on Every Retry

```python
response = await self.client.build_request(...)
if response.status_code in RETRYABLE_STATUS_CODES:
    if attempt < MAX_RETRIES:
        delay = 2 ** attempt
        _LOGGER.warning("Retrying in %ds after HTTP %d", delay, response.status_code)
        await asyncio.sleep(delay)
        continue  # ← response body NEVER consumed or closed!
```

On every retry, the `response` object from the failed request is discarded without calling `await response.aclose()`. The `httpx` response holds an open connection until garbage-collected. Under high retry load, this exhausts the connection pool.

**Fix:** Consume and close the response before every `continue`:
```python
if response.status_code in RETRYABLE_STATUS_CODES:
    if attempt < MAX_RETRIES:
        await response.aread()   # consume body
        await response.aclose()  # release connection
        delay = 2 ** attempt
        await asyncio.sleep(delay)
        continue
```

---

### A.3 🚨 BUG-05: Self-Referential Exception Chain in `_async_request_with_retry`

**File:** `client.py` lines 475–483  
**Severity:** 🔴 CRITICAL — Exception Chain Corruption

```python
except Exception as err:
    error_detail = response_text if response_text is not None else str(err)
    if attempt == MAX_RETRIES:
        raise VeniceAIError(
            f"Venice AI API request failed after {MAX_RETRIES} retries: {error_detail}"
        ) from err  # ← err is the same exception being raised in some paths
```

In the final attempt, if the exception raised is the same type as `err`, the `from err` creates a self-referential chain. More critically, the `else` block at line 483 is **unreachable**:

```python
else:
    return response
```

This `else` belongs to the `for` loop and executes only when the loop completes without `break`/`continue`/`return`. But the loop body always ends with either `continue`, `return`, or `raise` — so `else` never runs.

**Fix:** Remove the unreachable `else` block. Fix the exception chain:
```python
except Exception as err:
    error_detail = response_text if response_text is not None else str(err)
    raise VeniceAIError(
        f"API request failed after {MAX_RETRIES} retries: {error_detail}"
    ) from err
```

---

### A.4 🟠 ISSUE-04: `return False` for Auth Errors Instead of `ConfigEntryAuthFailed`

**File:** `__init__.py` lines 134–139  
**Severity:** 🟠 HIGH

```python
except AuthenticationError as err:
    _LOGGER.error("Authentication failed: %s", err)
    return False
```

Returning `False` from `async_setup_entry` marks the config entry as "failed to setup" and triggers HA's generic retry logic. The user sees a cryptic error. The correct pattern is to raise `ConfigEntryAuthFailed`, which triggers HA's re-authentication flow.

**Fix:**
```python
except AuthenticationError as err:
    raise ConfigEntryAuthFailed("Invalid API key") from err
```

---

### A.5 🟠 ISSUE-05: Deprecated `async_forward_entry_setups` Usage

**File:** `__init__.py` line 197  
**Severity:** 🟠 HIGH — Will break in future HA versions

```python
await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
```

This method's behavior has shifted across HA versions. In modern HA (2024.5+), the per-platform `async_forward_entry_setup` (singular) is the newer, recommended approach for integrations targeting recent versions.

**Fix:** Verify target HA version and use the appropriate API. For modern HA:
```python
for platform in PLATFORMS:
    await hass.config_entries.async_forward_entry_setup(entry, platform)
```

---

### A.6 🟠 ISSUE-06: Broad `Exception` Catches Mask Real Errors

**Files:** `__init__.py` line 134; `config_flow.py` lines 122, 232; `conversation.py` line 381; `ai_task.py` line 139; `stt.py` line 219  
**Severity:** 🟠 HIGH

Bare `except Exception` catches everything including `asyncio.CancelledError` and programming errors like `AttributeError`, re-wrapping them and making debugging much harder.

**Fix:** Catch specific expected exception types:
```python
except VeniceAIError as err:
    _LOGGER.error("Venice AI error: %s", err)
    raise HomeAssistantError(f"Venice AI error: {err}") from err
except (httpx.HTTPError, json.JSONDecodeError) as err:
    _LOGGER.exception("Unexpected transport/data error")
    raise HomeAssistantError(f"Communication error: {err}") from err
```

---

### A.7 🟠 ISSUE-07: No `async_reload_entry` Implementation

**File:** `__init__.py`  
**Severity:** 🟠 HIGH

When a user changes options (e.g., switches chat model, changes TTS voice), the options flow saves the new values but the entry is never reloaded. TTS and STT entities read options **only at `async_setup_entry` time** — changes won't take effect until manual reload or HA restart.

**Fix:** Add `async_reload_entry`:
```python
async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Reload Venice AI when options change."""
    await async_unload_entry(hass, entry)
    return await async_setup_entry(hass, entry)
```

And in `async_setup_entry`, register the listener:
```python
entry.async_on_unload(entry.add_update_listener(async_reload_entry))
```

---

### A.8 🟡 ISSUE-08: `VeniceAIConfigEntry` Subclasses `ConfigEntry` Incorrectly

**File:** `__init__.py` lines 44–47  
**Severity:** 🟡 MEDIUM — Anti-pattern, fragile to HA internal changes

```python
class VeniceAIConfigEntry(ConfigEntry):
    """Venice AI config entry with runtime data."""
    runtime_data: AsyncVeniceAIClient
```

`ConfigEntry` is a core HA class not designed to be subclassed by integrations. The `async_setup_entry` signature accepts `VeniceAIConfigEntry` but receives a plain `ConfigEntry` from HA's internals. At runtime, `entry.runtime_data = client` works because Python allows setting arbitrary attributes, but the subclass provides a false sense of type safety.

**Fix:** Remove the subclass. Use `ConfigEntry` directly:
```python
# Remove VeniceAIConfigEntry class entirely

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    ...
    entry.runtime_data = client  # Still works without subclass
```

---

### A.9 🟡 ISSUE-09: `tts.py` `async_stream_tts_audio` Is Not Truly Streaming

**File:** `tts.py` lines 143–159  
**Severity:** 🟡 MEDIUM — Misleading API, buffers entire message

```python
message = "".join([chunk async for chunk in request.message_gen])  # Buffers everything
options = dict(request.options or {})
audio_format, audio_data = await self.async_get_tts_audio(message, request.language, options)
...
async def gen() -> AsyncGenerator[bytes, None]:
    yield audio_data  # Yields a single chunk
return TTSAudioResponse(audio_format, gen())
```

The method name implies streaming, but it:
1. Collects ALL chunks into a single string
2. Generates the entire audio in one blocking call
3. Returns a generator that yields exactly one chunk

This is functionally identical to the non-streaming path.

**Fix:** Either implement true streaming if the API supports it, or don't override `async_stream_tts_audio` at all (HA will fall back to the non-streaming path).

---

### A.10 🟡 ISSUE-10: Hardcoded Model Parameters in `ai_task.py`

**File:** `ai_task.py` lines 98–104  
**Severity:** 🟡 MEDIUM

```python
response_data = await self._client.chat.create_non_streaming({
    "model": model,
    "messages": messages,
    "max_tokens": 1000,      # HARDCODED
    "temperature": 0.7,      # HARDCODED
    "stream": False,
})
```

`max_tokens` and `temperature` are hardcoded. Users cannot adjust AI task behavior.

**Fix:** Add to options or reuse conversation settings:
```python
max_tokens = self.entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
temperature = self.entry.options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
```

---

### A.11 🟡 ISSUE-11: Lazy Import Inside Hot Path in `stt.py`

**File:** `stt.py` line 150  
**Severity:** 🟡 MEDIUM

```python
async def async_process_audio_stream(self, ...):
    from .client import AsyncVeniceAIClient, VeniceAIError  # Lazy import
```

This import runs inside the most frequently-called STT method. There is no circular dependency preventing a top-level import.

**Fix:** Move to top-level imports.

---

### A.12 🟡 ISSUE-12: `stt.py` Accepts `hass` Parameter But Never Uses It

**File:** `stt.py` lines 84–91  
**Severity:** 🟡 MEDIUM

```python
def __init__(self, hass: HomeAssistant, entry: ConfigEntry, model: str, ...):
```

The `hass` parameter is accepted but never assigned to `self.hass` or used. `SpeechToTextEntity` already provides `self.hass` via the entity base class.

**Fix:** Remove the `hass` parameter from `__init__` and update the caller in `async_setup_entry`.

---

### A.13 🟡 ISSUE-13: `RECOMMENDED_MAX_TOKENS = 150` Is Extremely Low

**File:** `const.py` line 10  
**Severity:** 🟡 MEDIUM

```python
RECOMMENDED_MAX_TOKENS = 150
```

150 tokens is roughly 100–120 words. For a conversation AI that needs to return tool calls, explanations, and multi-sentence responses, this is severely restrictive. Most conversation agents default to 256–1024 tokens.

**Fix:** Increase to at least 512:
```python
RECOMMENDED_MAX_TOKENS = 512
```

Also update `config_flow.py` line 263 `NumberSelectorConfig` max from `max=4096` to match the model's actual context window.

---

### A.14 🔵 ISSUE-14: Duplicate `voluptuous_openapi` Detection

**Files:** `__init__.py` (lines 32–36) and `conversation.py` (lines 10–14)  
**Severity:** 🔵 LOW

Both files independently check for `voluptuous_openapi` availability. This should be centralized in `const.py`.

**Fix:** Export `HAS_VOLUPTUOUS_OPENAPI` from `const.py` once.

---

### A.15 🔵 ISSUE-15: Inconsistent Logger Variable Naming in `tts.py`

**File:** `tts.py` line 37  
**Severity:** 🔵 LOW

```python
_LOGGER = logging.getLogger(__name__)
# Extra blank line after logger
```

All other files follow the pattern of defining `_LOGGER` immediately after imports without a blank line.

---

### A.16 🔵 ISSUE-16: `_format_venice_schema` Loses Selector Type Information

**File:** `conversation.py` lines 79–107  
**Severity:** 🔵 LOW

```python
if isinstance(obj, selector.Selector):
    return str
```

All selector-based parameters (entity selectors, device selectors, area selectors) are flattened to plain strings. The AI receives no information about valid entity IDs or area names.

**Fix:** Preserve selector metadata where possible:
```python
if isinstance(obj, selector.EntitySelector):
    return {"type": "string", "description": "A Home Assistant entity ID"}
```

---

### A.17 🔵 ISSUE-17: `import ulid as ulid_util` Is Unconventional

**File:** `conversation.py` line 54  
**Severity:** 🔵 LOW

```python
from homeassistant.util import ulid as ulid_util
```

The alias `ulid_util` adds no clarity. Just use `import ulid` and call `ulid.ulid_now()`.

**Fix:**
```python
from homeassistant.util import ulid
# usage: ulid.ulid_now()
```

---

### A.18 🔵 ISSUE-18: `_async_entry_updated` Callback Does Nothing Useful

**File:** `conversation.py` lines 409–412  
**Severity:** 🔵 LOW

```python
def _async_entry_updated(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    self.entry = entry
```

This reassigns `self.entry`, which is redundant if `async_reload_entry` is implemented (ISSUE-07). TTS/STT entities never see option updates without reload.

**Fix:** Remove this callback and rely on `async_reload_entry` (ISSUE-07).

---

### A.19 🔵 ISSUE-19: TTS Error Message Leaks User Message Content

**File:** `tts.py` line 154  
**Severity:** 🔵 LOW

```python
raise HomeAssistantError(f"No TTS from {self.entity_id} for '{message}'")
```

Error messages include the full user text, which may be long or contain private information.

**Fix:**
```python
raise HomeAssistantError(f"No TTS audio generated from {self.entity_id}")
```

---

### A.20 🔵 ISSUE-20: STT Format Validation Is Verbose and Repetitive

**File:** `stt.py` lines 152–182  
**Severity:** 🔵 LOW

Five nearly identical `if metadata.X not in self.supported_Y` blocks (30 lines that could be 10).

**Fix:** Use a validation loop:
```python
validations = [
    (metadata.format, self.supported_formats, "format"),
    (metadata.codec, self.supported_codecs, "codec"),
    (metadata.bit_rate, self.supported_bit_rates, "bit rate"),
    (metadata.sample_rate, self.supported_sample_rates, "sample rate"),
    (metadata.channel, self.supported_channels, "channel count"),
]
for value, supported, label in validations:
    if value not in supported:
        _LOGGER.error("Unsupported audio %s: %s", label, value)
        return stt.SpeechResult("", stt.SpeechResultState.ERROR)
```

---

### A.21 🔵 ISSUE-21: Options Flow Complex `try/except/else` Flow

**File:** `config_flow.py` lines 170–242  
**Severity:** 🔵 LOW

The `async_step_init` has deeply nested `try/except/else` logic that handles both "client created successfully but no models" and "client was None" in the same `else` block. This is hard to follow.

**Fix:** Refactor model fetching into a helper method.

---

### A.22 💧 LEAK-02: `create_non_streaming` Mutates Input `payload` Dict

**File:** `client.py` lines 105–148  
**Severity:** 💧 LOW

```python
payload["stream"] = False  # Mutates caller's dict!
```

If the caller reuses the payload dict, this side effect causes bugs.

**Fix:** Copy the payload:
```python
payload = {**payload, "stream": False}
```

---

### A.23 💧 LEAK-03: No `weakref` or Cleanup for Conversation Entity Tool Results

**File:** `conversation.py` lines 313–361  
**Severity:** 💧 LOW

The `chat_log.content` list is appended to during each tool call iteration. For long conversations with many tool calls, this list grows unbounded. A malicious or buggy tool loop could append thousands of entries.

**Fix:** Add a hard limit on chat_log.content length:
```python
MAX_CHAT_LOG_LENGTH = 200
if len(chat_log.content) > MAX_CHAT_LOG_LENGTH:
    _LOGGER.warning("Chat log exceeds %d entries, truncating", MAX_CHAT_LOG_LENGTH)
    chat_log.content = chat_log.content[-MAX_CHAT_LOG_LENGTH:]
```

---

### A.24 💀 DEAD-01: `UNSUPPORTED_MODELS = []` in `const.py`

**File:** `const.py` line 45  
**Severity:** 💀 LOW

```python
UNSUPPORTED_MODELS = []
```

Never referenced anywhere. Remove it.

---

### A.25 💀 DEAD-02: `CONF_RECOMMENDED = "recommended"` in `const.py`

**File:** `const.py` line 5  
**Severity:** 💀 LOW

```python
CONF_RECOMMENDED = "recommended"
```

Not imported or used in any file. Remove it.

---

### A.26 💀 DEAD-03: `services.yaml` Referenced in CODE_REVIEW.md but Missing

The `CODE_REVIEW.md` references `services.yaml` in the file summary table. This file does not exist.

**Fix:** Update `CODE_REVIEW.md` or create the file.

---

### A.27 💀 DEAD-04: `_make_schema_hashable` Name Is Misleading

**File:** `conversation.py` lines 64–76  
**Severity:** 💀 LOW

The function exists only to prepare schemas for `voluptuous_openapi.convert()`. The name suggests general hashability.

**Fix:** Rename to `_prepare_schema_for_voluptuous_openapi` with a descriptive docstring.

---

### A.28 🏆 PLAT-04: No `repairs.py` for Repair Flows

**Severity:** 🏆 PLATINUM

Integration should handle common failure modes (expired API key, API deprecation, model removal) with HA's repair system.

---

### A.29 🏆 PLAT-07: No Strict Typing or `py.typed` Marker

**Severity:** 🏆 PLATINUM

Issues found:
- `conversation.py` line 221: `context: Any = None` should use proper type
- `tts.py` line 107: `options: dict[str, Any] | None = None` but body mutates with `options = {}`

**Fix:** Add `py.typed` marker file and complete type annotations.

---

### A.30 🏆 PLAT-08: Missing `RECOMMENDED_*` Constants for AI Task

**Severity:** 🏆 PLATINUM

`ai_task.py` hardcodes values that belong in `const.py`:
- `AI_TASK_MAX_TOKENS = 1000`
- `AI_TASK_TEMPERATURE = 0.7`

---

### A.31 🏆 PLAT-09: No Rate Limiting Awareness in Conversation Tool Loop

**Severity:** 🏆 PLATINUM

The tool loop can make up to `MAX_TOOL_ITERATIONS` (5) API calls rapidly. If Venice AI has rate limits, this could trigger 429 errors. The retry logic helps, but proactive rate limiting would be better.

---

### A.32 🏆 PLAT-10: `strings.json` Services Description Quality

**File:** `strings.json`  
**Severity:** 🏆 PLATINUM

```json
"ai_task": {
    "name": "VeniceAI AI Task",
    "description": "AI task"
}
```

The description "AI task" is circular and unhelpful.

**Fix:**
```json
"description": "Generate structured data or text using Venice AI based on a task description."
```

---

### A.33 🏗️ ARCH-01: Overlapping Device Info Model Names

**Severity:** 🏗️ ARCHITECTURAL

All four platforms register devices with the same identifier but different `model` fields. Only the last entity to register persists its `model` value.

**Fix:** Use a consistent generic model name: `model="Venice AI"` across all platforms.

---

### A.34 🏗️ ARCH-02: Client Shared Across All Platforms

**Severity:** 🏗️ ARCHITECTURAL

The single `AsyncVeniceAIClient` instance is shared by conversation, TTS, STT, and AI task. This means:
- A TTS request can be blocked by a long conversation
- If one platform closes the client, all break

This is acceptable for sequential usage but should be documented.

---

### A.35 🏗️ ARCH-03: Services Registered in `async_setup()` (Global Scope)

**File:** `__init__.py`  
**Severity:** 🏗️ ARCHITECTURAL

The `generate_image` and `ai_task` services are registered in `async_setup()`, which runs once globally. They're available even if no config entry exists. The service handlers access `entry.runtime_data` which won't exist.

**Fix:** Add early validation in service handlers:
```python
if entry is None:
    raise ServiceValidationError("No Venice AI integration configured")
```

Or register services only after an entry is set up.

---

*End of Analysis. This document should be treated as a living document and updated as fixes are applied.*

---

## Implementation Status Update — 2026-05-01

### ✅ Phase 1: Runtime Stability — COMPLETED

The following critical bugs have been fixed and all modified files pass `python -m py_compile`:

| Fix | File(s) | Status | Notes |
|---|---|---|---|
| **BUG #1** — `ai_task` Import Failure | `__init__.py`, `ai_task.py` | ✅ Fixed | Added `try/except ImportError` guard. `PLATFORMS` list and `SERVICE_AI_TASK` registration are now conditional. `ai_task.py` defines a dummy class when unavailable. |
| **BUG #2** — `create()` signature mismatch | `conversation.py` | ✅ Fixed | Replaced streaming `create()` call with `create_non_streaming(payload)` using a dict payload. |
| **2.1** — Streaming response leak | `client.py` | ✅ Fixed | Wrapped `create()` with `@asynccontextmanager`; inner generator calls `response.aclose()` in `finally`. |
| **2.3 / BUG-04** — Response body not consumed before retry | `client.py` | ✅ Fixed | Added `await response.aread()` before retry `continue` in `_async_request_with_retry()`. |
| **BUG-05** — Self-referential exception chain | `client.py` | ✅ Fixed | Replaced `raise last_err from err` with `raise VeniceAIError(...)` and added explicit final `raise`. |
| **2.2** — Response validation in service handlers | `__init__.py` | ✅ Fixed | Added `isinstance(response, dict)` guard in `render_image` service handler. |
| **A.4** — `ConfigEntryAuthFailed` instead of `return False` | `__init__.py` | ✅ Fixed | `AuthenticationError` now raises `ConfigEntryAuthFailed`. |
| **7.1** — `hass.data[DOMAIN]` cleanup | `__init__.py` | ✅ Fixed | `async_unload_entry` now pops entry ID and removes empty dict. |
| **BUG-03** — Client leak in options flow | `config_flow.py` | ✅ Fixed | Added `finally` block to close `_client` after model fetching. |
| **Pylance errors** — Missing return / type hints | `client.py` | ✅ Fixed | Added explicit `raise` for unreachable path and corrected return types. |

### 📋 Remaining Work (Phases 2–4)

#### Phase 2: Reliability
- **3.1** — Expand `supported_languages` in `conversation.py`
- **3.2** — Add `DataUpdateCoordinator` (`coordinator.py`)
- **3.3** — Ensure `AsyncVeniceAIClient` close safety in direct instantiation
- **3.4** — Validate `entry.runtime_data` type in `ai_task.py` & `conversation.py`
- **4.1** — Implement re-auth flow in `config_flow.py`
- **4.3** — Handle empty audio in `stt.py`
- **4.5** — Make `MAX_TOOL_ITERATIONS` configurable
- **7.2** — Trim `ChatLog` content to prevent unbounded growth

#### Phase 3: Platinum Compliance
- **6.2** — Add `diagnostics.py` platform
- **6.3** — Add `quality_scale.yaml`
- **6.4** — Complete `strings.json` translations for all error keys
- **A.7** — Implement `async_reload_entry`
- **A.9** — Fix TTS streaming (either true streaming or remove override)
- **A.10** — Remove hardcoded params from `ai_task.py`
- **A.11** — Move lazy import to top-level in `stt.py`
- **A.12** — Remove unused `hass` param from `stt.py` `__init__`
- **A.13** — Increase `RECOMMENDED_MAX_TOKENS` from 150
- **A.29** — Add `py.typed` marker
- Add comprehensive tests

#### Phase 4: Polish
- **5.1** — Reduce `voluptuous_openapi` log noise
- **5.3** — Clean up redundant "Venice AI" prefixes in exceptions
- **8.1** — Use or remove `_HAS_VOLUPTUOUS_OPENAPI`
- **8.3** — Remove redundant `response_text` pattern in `client.py`
- **8.4** — Implement or remove `disable_thinking` option
- **A.14** — Centralize `voluptuous_openapi` detection in `const.py`
- **A.16** — Preserve selector metadata in `_format_venice_schema`
- **A.17** — Clean up `ulid` import alias
- **A.18** — Remove `_async_entry_updated` or make it useful
- **A.19** — Remove user message from TTS error strings
- **A.20** — Refactor STT format validation loop
- **A.21** — Refactor options flow into helper methods
- **A.22** — Copy payload dict before mutating in `create_non_streaming`
- **A.23** — Add `MAX_CHAT_LOG_LENGTH` guard
- **A.24–A.27** — Remove dead code (`UNSUPPORTED_MODELS`, `CONF_RECOMMENDED`, unused `services.yaml` ref, rename `_make_schema_hashable`)
- **A.28** — Add `repairs.py` for repair flows
- **A.33** — Standardize device info model names
- **A.34** — Document client-sharing architecture decision
- **A.35** — Validate config entry existence in service handlers

### 📝 Dev Strategies & Lessons Learned (2026-05-01)

1. **Conditional platform imports require module-level guards in TWO places**
   - `__init__.py` must conditionally include the platform in `PLATFORMS` and service registration.
   - The platform module itself must be import-safe (define dummy classes when imports fail).
   - Without both guards, Pylance/runtime errors occur on HA versions lacking the platform.

2. **Home Assistant's `ConfigEntryAuthFailed` is the correct pattern for auth errors**
   - Returning `False` from `async_setup_entry` triggers generic retry loops.
   - `raise ConfigEntryAuthFailed(...)` triggers HA's built-in re-authentication UI flow.

3. **`@asynccontextmanager` is the safest way to manage httpx streaming responses**
   - Python async generators' `finally` blocks are unreliable on cancellation.
   - `@asynccontextmanager` guarantees `response.aclose()` even on exceptions or cancellation.

4. **httpx connection pool exhaustion is subtle — always consume response bodies before retry**
   - `await response.aread()` is required before discarding a response object.
   - Without this, retries silently exhaust the pool and subsequent requests hang.

5. **Self-referential exception chains (`raise e from e`) corrupt tracebacks**
   - Python's `from` clause should always chain to a *different* exception or be omitted.
   - Re-raising with a wrapper exception type avoids the circular reference.

6. **The `else` block on `for` loops only runs on natural completion**
   - If the loop body always exits via `return`, `raise`, or `continue`, the `else` is dead code.
   - This pattern caused a "missing return" Pylance error that was actually an unreachable branch.

7. **SEARCH/REPLACE on large documents is fragile**
   - The `deep-dive-analysis.md` file is 1951 lines; the `replace_in_file` tool requires exact character matches.
   - For large docs, append new sections at the end rather than modifying inline checkboxes.
