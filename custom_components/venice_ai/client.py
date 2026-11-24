"""Venice AI API Client."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx

_LOGGER = logging.getLogger(__name__)

# Import retry / timeout constants; fall back to hard-coded defaults when
# const.py is not yet importable (e.g. during isolated unit tests).
try:
    from .const import (
        MAX_RETRIES,  # MED-4
        RETRY_BASE_DELAY,
        RETRY_MAX_DELAY,
        DEFAULT_HTTP_TIMEOUT,  # QUAL-2: tunable per-request timeout default.
        DEFAULT_HTTP_KEEPALIVE,  # QUAL-2: connection-pool sizing.
        DEFAULT_HTTP_MAX_CONNECTIONS,  # QUAL-2: connection-pool sizing.
        DEFAULT_CHAT_TIMEOUT,
        DEFAULT_CHAT_STREAM_TIMEOUT,
        DEFAULT_TTS_TIMEOUT,
        DEFAULT_STT_TIMEOUT,
        DEFAULT_IMAGE_TIMEOUT,
    )
except ImportError:  # pragma: no cover
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0
    RETRY_MAX_DELAY = 30.0
    DEFAULT_HTTP_TIMEOUT = 30.0
    DEFAULT_HTTP_KEEPALIVE = 5
    DEFAULT_HTTP_MAX_CONNECTIONS = 10
    DEFAULT_CHAT_TIMEOUT = 120.0
    DEFAULT_CHAT_STREAM_TIMEOUT = 300.0
    DEFAULT_TTS_TIMEOUT = 60.0
    DEFAULT_STT_TIMEOUT = 60.0
    DEFAULT_IMAGE_TIMEOUT = 120.0


def _sanitize_header_value(value: str | None) -> str:
    """SEC-1: strip CR/LF from a header value before it goes on the wire.

    Only the carriage-return and line-feed bytes are removed. We deliberately
    do NOT:

      * call ``.strip()`` — that would silently mutate a credential by
        trimming leading/trailing whitespace (or NBSP/other Python-defined
        whitespace), producing a byte-different key that authenticates
        against Venice as "invalid" (regression seen in commit 64b115c,
        which caused HTTP 401 on previously-valid keys).
      * filter every char below 0x20 — Venice API keys are alphanumeric
        today, but filtering by ord >= 0x20 plus a trailing .strip() was
        the exact combination that broke valid keys. Header injection is
        fully prevented by removing just ``\\r`` and ``\\n``; httpx itself
        rejects any remaining control bytes on the wire.

    The original, unmodified ``api_key`` is stored on ``self._api_key``;
    this function is only applied at header-construction time so the
    Authorization value is the credential byte-for-byte minus CR/LF.
    """
    if not value:
        return ""
    return value.replace("\r", "").replace("\n", "")


# PERF-1: process-wide cache of model lists. Multiple client instances (e.g.
# across config entries or per-test harnesses) reuse a single result for
# _PROCESS_MODEL_CACHE_TTL. The cache stores plain dicts independently of any
# specific httpx client lifetime.
_PROCESS_MODEL_CACHE: dict[tuple[str, bytes, str], tuple[float, list[dict]]] = {}
_PROCESS_MODEL_CACHE_TTL = 3600
_MODEL_CACHE_GENERATION: dict[tuple[str, bytes, str], int] = {}


@dataclass
class VeniceAIMetrics:
    """Lightweight in-memory usage/telemetry counters for a client instance.

    These counters back the diagnostic sensor entities (LOW-4) so users can
    monitor API usage, token consumption, and error rates without enabling
    debug logging. All counters are cumulative for the lifetime of the client
    (i.e. until the config entry is reloaded).
    """

    request_count: int = 0
    error_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    last_error: str | None = None
    _listeners: list[Callable[[], None]] = field(
        default_factory=list, repr=False, compare=False
    )

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Register a callback invoked after a counter changes.

        The returned callable removes that listener. A listener that raises is
        logged and does not fail the API call that recorded the metric.
        """
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                return

        return _unsubscribe

    def _notify_listeners(self) -> None:
        """Tell subscribers that a counter changed."""
        for listener in tuple(self._listeners):
            try:
                listener()
            except Exception:
                _LOGGER.exception("Venice AI metrics listener failed")

    def record_request(self) -> None:
        """Increment the total request counter."""
        self.request_count += 1
        self._notify_listeners()

    def record_error(self, error: BaseException) -> None:
        """Increment the error counter and remember the last error message."""
        self.error_count += 1
        self.last_error = f"{type(error).__name__}: {error}"
        self._notify_listeners()

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """Accumulate token usage from an API ``usage`` block, if present."""
        if not isinstance(usage, dict):
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        self._notify_listeners()



class VeniceAIError(Exception):
    """Base exception for Venice AI errors."""


class AuthenticationError(VeniceAIError):
    """Authentication error (HTTP 401)."""


class RateLimitError(VeniceAIError):
    """Rate-limit error (HTTP 429) — the Venice AI API quota has been exceeded."""


class ServiceUnavailableError(VeniceAIError):
    """Service unavailable error (HTTP 5xx) — Venice AI is temporarily down."""


class NetworkError(VeniceAIError):
    """Network-level error — could not reach the Venice AI API (timeout, connection refused, etc.)."""


def chat_completion_payload(data: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    """Build a chat-completion body Venice will accept.

    Null fields are omitted. Venice rejects ``tools: null`` and the other
    optional keys with HTTP 400 "Invalid request parameters". A whole-number
    ``max_tokens`` is sent as an integer; a float fails the same check.
    """
    payload = {key: value for key, value in data.items() if value is not None}
    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, float) and max_tokens.is_integer():
        payload["max_tokens"] = int(max_tokens)
    payload["stream"] = stream
    return payload


def api_error_message(error_detail: str) -> str:
    """Return the API's message, including field details when present."""
    if not error_detail:
        return error_detail
    try:
        error_json = json.loads(error_detail)
    except json.JSONDecodeError:
        return error_detail
    if not isinstance(error_json, dict):
        return error_detail
    error = error_json.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or error_detail)
    elif isinstance(error, str) and error:
        message = error
    else:
        message = error_detail
    details = error_json.get("details")
    if details:
        return f"{message}: {details}"
    return message


def _categorize_http_error(
    status_code: int, error_detail: str, context: str = ""
) -> VeniceAIError:
    """Return the most specific VeniceAIError subtype for an HTTP status code.

    Centralises status-code → exception-type mapping so every API method raises
    a consistent, typed exception.  Callers should ``raise ... from err`` the
    returned instance directly.

    Args:
        status_code: The HTTP response status code.
        error_detail: A human-readable description (from the response body).
        context: Optional description of the operation, e.g. ``"fetching models"``.
    """
    suffix = f" ({context})" if context else ""
    msg = f"HTTP error {status_code}{suffix}: {error_detail}"
    if status_code == 401:
        return AuthenticationError(f"Invalid API key{suffix}")
    if status_code == 429:
        return RateLimitError(msg)
    if status_code >= 500:
        return ServiceUnavailableError(msg)
    return VeniceAIError(msg)


class ChatCompletionChunk:
    """Chat completion chunk (used for streaming responses)."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize chat completion chunk."""
        self.choices = data.get("choices", [])
        # Populated in the final chunk when stream_options.include_usage=True
        self.usage: dict[str, Any] | None = data.get("usage")


class ChatCompletions:
    """Chat completions API for Venice AI."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize chat completions."""
        self.client = client
        # Allow both client.chat.create_non_streaming(...) and
        # client.chat.completions.create_non_streaming(...) — the latter mirrors
        # the OpenAI SDK's chat.completions namespace and is the preferred form.
        self.completions = self

    @asynccontextmanager
    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        venice_parameters: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncGenerator[AsyncGenerator[ChatCompletionChunk, None], None]:
        """Create a streaming chat completion."""
        data = chat_completion_payload(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "venice_parameters": venice_parameters,
                "tools": tools,
                "tool_choice": tool_choice,
                "stream_options": stream_options,
                "response_format": response_format,
            },
            stream=True,
        )

        self.client.metrics.record_request()
        response: httpx.Response | None = None
        _connect_start = time.monotonic()
        _LOGGER.debug("[PERF-HTTP] POST /chat/completions (stream) — opening connection")
        try:
            request = self.client._http_client.build_request(
                "POST",
                f"{self.client._base_url}/chat/completions",
                headers=self.client._headers,
                json=data,
                timeout=DEFAULT_CHAT_STREAM_TIMEOUT,
            )
            response = await self.client._http_client.send(request, stream=True)
            response.raise_for_status()
            _LOGGER.debug(
                "[PERF-HTTP] POST /chat/completions (stream) → HTTP %d, connection established in %.3fs",
                response.status_code,
                time.monotonic() - _connect_start,
            )

        except httpx.HTTPStatusError as err:
            if response is not None:
                await response.aclose()
            error_detail = ""
            try:
                error_detail = err.response.text
            except Exception:
                pass
            _LOGGER.error("Venice AI HTTP error %s: %s", err.response.status_code, error_detail)
            categorized = _categorize_http_error(
                err.response.status_code, error_detail, "streaming chat"
            )
            self.client.metrics.record_error(categorized)
            raise categorized from err
        except httpx.RequestError as err:
            if response is not None:
                await response.aclose()
            _LOGGER.error("Venice AI request error: %s", err)
            network_err = NetworkError(f"Request error (streaming chat): {err}")
            self.client.metrics.record_error(network_err)
            raise network_err from err

        async def _stream() -> AsyncGenerator[ChatCompletionChunk, None]:
            _chunk_count = 0
            _stream_start = time.monotonic()
            try:
                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk_data = json.loads(line[6:])
                            _chunk_count += 1
                            chunk = ChatCompletionChunk(chunk_data)
                            # Final chunk carries usage when include_usage is set.
                            # Non-streaming already records this; streaming must too
                            # or the diagnostic sensors stay at 0 (the default path).
                            if chunk.usage:
                                self.client.metrics.record_usage(chunk.usage)
                            yield chunk
                        except json.JSONDecodeError:
                            _LOGGER.warning("Failed to decode stream chunk: %s", line)
                    else:
                        _LOGGER.warning("Received unexpected line in stream: %s", line)
                _LOGGER.debug(
                    "[PERF-HTTP] POST /chat/completions stream finished — %d chunks in %.3fs",
                    _chunk_count,
                    time.monotonic() - _stream_start,
                )
            except httpx.TransportError as err:
                # Convert mid-stream transport failures (connection reset, server close,
                # timeout) to a typed NetworkError so callers get consistent exceptions.
                _LOGGER.error("Stream interrupted by transport error: %s", err)
                network_err = NetworkError(f"Stream interrupted: {err}")
                self.client.metrics.record_error(network_err)
                raise network_err from err
            finally:
                if response is not None:
                    await response.aclose()

        yield _stream()

    async def create_non_streaming(
        self,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a non-streaming chat completion.

        Accepts either a pre-built payload dict (legacy positional form) or
        keyword arguments matching the OpenAI chat-completions API parameters
        (e.g. model=, messages=, max_tokens=, temperature=, top_p=, tools=).
        Both calling conventions are equivalent:

            # dict form (legacy):
            await client.chat.completions.create_non_streaming(
                {"model": "...", "messages": [...]}
            )

            # keyword form (preferred, mirrors OpenAI SDK):
            await client.chat.completions.create_non_streaming(
                model="...", messages=[...]
            )
        """
        if payload is None:
            payload = {}
        # Merge keyword arguments; kwargs take precedence over dict keys.
        if kwargs:
            payload = {**payload, **kwargs}
        payload = chat_completion_payload(payload, stream=False)

        self.client.metrics.record_request()
        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/chat/completions",
                headers=self.client._headers,
                json=payload,
                timeout=DEFAULT_CHAT_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            # Accumulate token usage for the diagnostic sensors (LOW-4).
            if isinstance(result, dict):
                self.client.metrics.record_usage(result.get("usage"))
            return result

        except httpx.HTTPStatusError as err:
            error_detail = getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI HTTP error %s: %s", err.response.status_code, error_detail)
            error_message = api_error_message(error_detail)
            categorized = _categorize_http_error(err.response.status_code, error_message, "chat completion")
            self.client.metrics.record_error(categorized)
            raise categorized from err

        except httpx.RequestError as err:
            _LOGGER.error("Venice AI request error: %s", err)
            network_err = NetworkError(f"Request error (chat completion): {err}")
            self.client.metrics.record_error(network_err)
            raise network_err from err

        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode non-streaming JSON response: %s", response.text)
            decode_err = VeniceAIError(f"Failed to decode API response: {response.text}")
            self.client.metrics.record_error(decode_err)
            raise decode_err from err



class Models:
    """Models API for Venice AI with TTL caching."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize models API."""
        self.client = client

    async def list(
        self, model_type: str = "text", *, force_refresh: bool = False
    ) -> list[dict]:
        """List available models with TTL caching.

        Cached results are scoped to the endpoint, credential and model type.
        Force a live request when validating credentials.
        """
        now = time.monotonic()
        key = (
            self.client._base_url,
            hashlib.sha256(self.client._api_key.encode()).digest(),
            model_type,
        )
        if force_refresh:
            _PROCESS_MODEL_CACHE.pop(key, None)
            _MODEL_CACHE_GENERATION[key] = _MODEL_CACHE_GENERATION.get(key, 0) + 1
        generation = _MODEL_CACHE_GENERATION.get(key, 0)
        global_cached = _PROCESS_MODEL_CACHE.get(key)
        if global_cached is not None:
            timestamp, models = global_cached
            if now - timestamp < _PROCESS_MODEL_CACHE_TTL:
                _LOGGER.debug(
                    "PERF-1: process-wide cache hit for %s models (%d entries)",
                    model_type,
                    len(models),
                )
                return copy.deepcopy(models)

        url = f"{self.client._base_url}/models"
        _LOGGER.debug("Attempting to fetch %s models from URL: %s", model_type, url)
        try:
            response = await self.client._async_request_with_retry(
                "GET",
                "/models",
                headers=self.client._headers,
                params={"type": model_type},
            )
            response.raise_for_status()
            model_data = response.json()
            models = model_data.get("data", [])
            # An older in-flight request must not undo a forced validation.
            if _MODEL_CACHE_GENERATION.get(key, 0) == generation:
                _PROCESS_MODEL_CACHE[key] = (time.monotonic(), copy.deepcopy(models))
            _LOGGER.debug("Successfully fetched %d %s models", len(models), model_type)
            return models
        except httpx.HTTPStatusError as err:
            error_detail = getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI Models API HTTP error %s: %s", err.response.status_code, error_detail)
            raise _categorize_http_error(err.response.status_code, error_detail, "fetching models") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Models API request error: %s (URL: %s, type: %s)", err, url, type(err).__name__)
            raise NetworkError(f"Request error fetching models: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode models JSON response: %s", response.text)
            raise VeniceAIError("Failed to decode models API response") from err


SPEECH_ERROR_FIELD_LIMIT = 200
SPEECH_ERROR_BODY_LIMIT = 500
_SPEECH_REDACTION_MIN_INPUT = 4


def _decode_speech_body(raw: bytes | None) -> str:
    """Decode a speech error body, dropping NUL padding."""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").strip().replace("\x00", "")


def _cap_speech_field(
    value: str, limit: int = SPEECH_ERROR_FIELD_LIMIT
) -> str:
    """Cap a logged error field so a long body cannot flood the log."""
    value = value.strip()
    return value if len(value) <= limit else f"{value[:limit]}..."


def _redact_speech_input(value: str, input_text: str | None) -> str:
    """Replace the spoken input anywhere Venice echoes it back.

    Short inputs are skipped: replacing a common word such as "the" would
    make the remaining error unreadable without protecting anything. Both the
    plain and the JSON-escaped form are replaced so the raw body string is
    covered as well as parsed fields.
    """
    if not input_text:
        return value
    spoken = input_text.strip()
    if len(spoken) < _SPEECH_REDACTION_MIN_INPUT:
        return value
    redacted = value.replace(spoken, "[redacted]")
    escaped = json.dumps(spoken)[1:-1]
    if escaped and escaped != spoken:
        redacted = redacted.replace(escaped, "[redacted]")
    return redacted


def _parse_speech_error_fields(
    text: str,
) -> tuple[str | None, str | None, str | None] | None:
    """Return ``(message, code, type)`` from a JSON error body, if parseable."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    message: object | None = None
    code: object | None = None
    error_type: object | None = None
    error = data.get("error")
    if isinstance(error, dict):
        message = (
            error.get("message")
            or error.get("detail")
            or error.get("description")
        )
        code = error.get("code")
        error_type = error.get("type")
    elif isinstance(error, str) and error:
        message = error
    if message is None:
        message = data.get("message") or data.get("detail")
    code = code or data.get("code")
    error_type = error_type or data.get("type")
    details = data.get("details")
    if details:
        detail_text = str(details)
        message = f"{message}: {detail_text}" if message else detail_text

    def _as_text(value: object | None) -> str | None:
        if value is None:
            return None
        text_value = str(value).strip()
        return text_value or None

    return _as_text(message), _as_text(code), _as_text(error_type)


def _speech_error_detail(
    raw: bytes | None,
    content_type: str,
    status: int,
    input_text: str | None = None,
) -> str:
    """Return a redacted, loggable speech error summary.

    Venice sometimes answers HTTP 400 with ``Content-Type: audio/*`` and a
    JSON body. Only the parsed ``message``, ``code``, and ``type`` fields are
    kept, each capped, so a body that echoes the spoken input cannot leak it
    into logs, exception messages, or the diagnostic sensor. Unparseable
    bodies fall back to a marker instead of raw bytes.
    """
    text = _decode_speech_body(raw)
    if not text:
        return f"HTTP {status}"
    fields = _parse_speech_error_fields(text)
    if fields is None:
        return f"HTTP {status} (unparseable body, {content_type or 'no content-type'})"

    processed = [
        _cap_speech_field(_redact_speech_input(value, input_text)) if value else None
        for value in fields
    ]
    message, code, error_type = processed
    parts = []
    if message:
        parts.append(message)
    if code:
        parts.append(f"code={code}")
    if error_type:
        parts.append(f"type={error_type}")
    if not parts:
        return f"HTTP {status} (unparseable body, {content_type or 'no content-type'})"
    return f"HTTP {status} " + " ".join(parts)


def _streaming_speech_should_retry(detail: str) -> bool:
    """Return True when a streaming 400 should be retried as one audio file."""
    lowered = detail.lower()
    if "streaming" in lowered:
        return True
    # The body was not a useful API message. Voice/format failures name
    # themselves; those must not be retried as if streaming were the cause.
    if lowered.startswith("http 400") and not any(
        token in lowered for token in ("voice", "format", "input", "model", "speed")
    ):
        return True
    return False


async def _read_speech_error(
    response: httpx.Response, input_text: str | None = None
) -> str:
    """Read a failed speech response, including a streaming body.

    The decoded body is logged at debug level *after* redaction, so the words
    being spoken never appear in the log even if Venice echoes them back.
    """
    status = response.status_code
    content_type = response.headers.get("content-type", "")
    try:
        raw = await response.aread()
    except Exception:
        return f"HTTP {status}"
    body = _decode_speech_body(raw)
    if body:
        _LOGGER.debug(
            "Venice AI Speech API error body (redacted): %s",
            _cap_speech_field(
                _redact_speech_input(body, input_text), SPEECH_ERROR_BODY_LIMIT
            ),
        )
    return _speech_error_detail(raw, content_type, status, input_text=input_text)


def _speech_speed(speed: object) -> float:
    """Coerce speed so a string from options is not sent as JSON text."""
    try:
        value = float(speed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    if value != value:
        return 1.0
    return min(4.0, max(0.25, value))


class Speech:
    """Speech API for Venice AI."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize speech API."""
        self.client = client

    async def generate(
        self,
        text: str,
        voice: str = "bm_daniel",
        model: str = "tts-kokoro",
        audio_output: str = "mp3",
        speed: float = 1.0,
    ) -> bytes:
        """Generate speech audio from text (non-streaming)."""
        speed_value = _speech_speed(speed)
        data = {
            "input": text,
            "model": model,
            "voice": voice,
            "response_format": audio_output,
            "speed": speed_value,
        }

        audio_headers = {
            "Authorization": self.client.authorization_header(),
            "Accept": "audio/*, application/json",
        }

        _gen_start = time.monotonic()
        self.client.metrics.record_request()
        _LOGGER.debug(
            "[PERF-HTTP] POST /audio/speech (non-streaming) — model=%s, voice=%s, format=%s, text=%d chars",
            model, voice, audio_output, len(text),
        )
        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/audio/speech",
                headers=audio_headers,
                json=data,
                timeout=DEFAULT_TTS_TIMEOUT,
            )
            response.raise_for_status()
            audio_data = response.content
            _gen_elapsed = time.monotonic() - _gen_start
            _bytes_per_sec = len(audio_data) / _gen_elapsed if _gen_elapsed > 0 else 0.0
            _LOGGER.debug(
                "[PERF-HTTP] POST /audio/speech (non-streaming) — complete: %d bytes in %.3fs (%.0f bytes/s)",
                len(audio_data), _gen_elapsed, _bytes_per_sec,
            )
            return audio_data

        except httpx.HTTPStatusError as err:
            error_detail = await _read_speech_error(err.response, input_text=text)
            _LOGGER.error(
                "Venice AI Speech API HTTP error %s (model=%s voice=%s format=%s): %s",
                err.response.status_code,
                model,
                voice,
                audio_output,
                error_detail,
            )
            categorized = _categorize_http_error(
                err.response.status_code, error_detail, "generating speech"
            )
            self.client.metrics.record_error(categorized)
            raise categorized from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Speech API request error: %s", err)
            network_err = NetworkError(f"Request error generating speech: {err}")
            self.client.metrics.record_error(network_err)
            raise network_err from err

    async def generate_streaming(
        self,
        text: str,
        voice: str = "bm_daniel",
        model: str = "tts-kokoro",
        audio_output: str = "mp3",
        speed: float = 1.0,
    ) -> AsyncGenerator[bytes, None]:
        """Generate speech audio from text with streaming chunks.

        The request is sent with ``stream=True``. The buffered retry helper
        would download the whole file before the first chunk is yielded.
        """
        speed_value = _speech_speed(speed)
        data = {
            "input": text,
            "model": model,
            "voice": voice,
            "response_format": audio_output,
            "speed": speed_value,
            "streaming": True,
        }

        audio_headers = {
            "Authorization": self.client.authorization_header(),
            "Accept": "audio/*, application/json",
        }
        self.client.metrics.record_request()
        response: httpx.Response | None = None
        _tts_stream_start = time.monotonic()
        _tts_first_chunk_t: float | None = None
        _tts_chunk_count = 0
        _tts_total_bytes = 0
        try:
            request = self.client._http_client.build_request(
                "POST",
                f"{self.client._base_url}/audio/speech",
                headers=audio_headers,
                json=data,
                timeout=DEFAULT_TTS_TIMEOUT,
            )
            response = await self.client._http_client.send(request, stream=True)
            response.raise_for_status()
            _LOGGER.debug(
                "[PERF-HTTP] POST /audio/speech (stream) — connection established, streaming chunks"
            )
            async for chunk in response.aiter_bytes():
                if _tts_first_chunk_t is None:
                    _tts_first_chunk_t = time.monotonic() - _tts_stream_start
                    _LOGGER.debug(
                        "[PERF-HTTP] POST /audio/speech (stream) — first audio chunk received in %.3fs",
                        _tts_first_chunk_t,
                    )
                _tts_chunk_count += 1
                _tts_total_bytes += len(chunk)
                yield chunk
            _LOGGER.debug(
                "[PERF-HTTP] POST /audio/speech (stream) — complete: %d chunks, %d bytes total in %.3fs",
                _tts_chunk_count,
                _tts_total_bytes,
                time.monotonic() - _tts_stream_start,
            )
        except httpx.TransportError as err:
            _LOGGER.error("Streaming TTS transport error: %s", err)
            network_err = NetworkError(f"Streaming TTS interrupted: {err}")
            self.client.metrics.record_error(network_err)
            raise network_err from err
        except httpx.HTTPStatusError as err:
            error_detail = await _read_speech_error(err.response, input_text=text)
            _LOGGER.error(
                "Venice AI Speech API HTTP error %s (model=%s voice=%s format=%s streaming=True): %s",
                err.response.status_code,
                model,
                voice,
                audio_output,
                error_detail,
            )
            if err.response.status_code == 400 and _streaming_speech_should_retry(
                error_detail
            ):
                _LOGGER.warning(
                    "Venice rejected streaming speech (%s); retrying as one audio file",
                    error_detail,
                )
                if response is not None:
                    await response.aclose()
                    response = None
                try:
                    audio = await self.generate(
                        text,
                        voice=voice,
                        model=model,
                        audio_output=audio_output,
                        speed=speed_value,
                    )
                except VeniceAIError:
                    _LOGGER.error(
                        "Non-streaming speech retry failed after streaming HTTP %s: %s",
                        err.response.status_code,
                        error_detail,
                    )
                else:
                    if audio:
                        yield audio
                    return
            categorized = _categorize_http_error(
                err.response.status_code, error_detail, "streaming speech"
            )
            self.client.metrics.record_error(categorized)
            raise categorized from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Speech API request error: %s", err)
            network_err = NetworkError(
                f"Request error generating streaming speech: {err}"
            )
            self.client.metrics.record_error(network_err)
            raise network_err from err
        finally:
            if response is not None:
                await response.aclose()


class Transcriptions:
    """Transcriptions API for Venice AI."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize transcriptions API."""
        self.client = client

    async def create(
        self,
        audio_data: bytes,
        model: str = "nvidia/parakeet-tdt-0.6b-v3",
        response_format: str = "json",
        timestamps: bool = False,
    ) -> dict[str, Any]:
        """Create a transcription from audio data."""
        files = {
            "file": ("audio.wav", audio_data, "audio/wav"),
        }
        data = {
            "model": model,
            "response_format": response_format,
            "timestamps": str(timestamps).lower(),
        }

        multipart_headers = {
            "Authorization": self.client.authorization_header(),
        }

        _stt_start = time.monotonic()
        self.client.metrics.record_request()
        _LOGGER.debug(
            "[PERF-HTTP] POST /audio/transcriptions — model=%s, format=%s, audio=%d bytes",
            model, response_format, len(audio_data),
        )
        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/audio/transcriptions",
                headers=multipart_headers,
                files=files,
                data=data,
                timeout=DEFAULT_STT_TIMEOUT,
            )
            response.raise_for_status()
            _stt_elapsed = time.monotonic() - _stt_start
            _LOGGER.debug(
                "[PERF-HTTP] POST /audio/transcriptions — complete in %.3fs (HTTP %d)",
                _stt_elapsed, response.status_code,
            )
            if response_format == "json":
                return response.json()
            else:
                return {"text": response.text}

        except httpx.HTTPStatusError as err:
            error_detail = getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI Transcriptions API HTTP error %s: %s", err.response.status_code, error_detail)
            categorized = _categorize_http_error(
                err.response.status_code, error_detail, "creating transcription"
            )
            self.client.metrics.record_error(categorized)
            raise categorized from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Transcriptions API request error: %s", err)
            network_err = NetworkError(f"Request error creating transcription: {err}")
            self.client.metrics.record_error(network_err)
            raise network_err from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode transcriptions JSON response: %s", response.text)
            decode_err = VeniceAIError("Failed to decode transcriptions API response")
            self.client.metrics.record_error(decode_err)
            raise decode_err from err


IMAGE_GENERATE_FORMAT = "webp"
IMAGE_MIN_EDGE = 1
IMAGE_MAX_EDGE = 1280


class ImageGenerateError(ValueError):
    """Caller input rejected before POST /image/generate."""


def _constraint_choices(value: Any) -> list[str]:
    """Return non-empty strings from a model constraint list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _positive_int(value: Any) -> int | None:
    """Return a positive int, or None if the value is not a whole number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    number = int(value)
    if number <= 0:
        return None
    return number


def image_model_constraints(model: dict[str, Any]) -> dict[str, Any]:
    """Return the constraints object for one catalog model, or {}."""
    spec = model.get("model_spec") or model.get("modelSpec") or {}
    if not isinstance(spec, dict):
        return {}
    constraints = spec.get("constraints") or {}
    if not isinstance(constraints, dict):
        return {}
    return constraints


def find_image_model(
    models: list[Any],
    model_id: str,
) -> dict[str, Any] | None:
    """Find a type=image catalog entry by id.

    Entries with no type are accepted, because the list is already
    fetched with ``?type=image``. Video and other types are skipped.
    """
    for model in models:
        if not isinstance(model, dict) or model.get("id") != model_id:
            continue
        model_type = model.get("type")
        if model_type not in (None, "", "image"):
            continue
        return model
    return None


def _check_edge(name: str, value: Any, divisor: int) -> None:
    """Reject a pixel edge the native image API would answer 400 for."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImageGenerateError(f"{name} must be a whole number.")
    if value < IMAGE_MIN_EDGE or value > IMAGE_MAX_EDGE:
        raise ImageGenerateError(
            f"{name} must be between {IMAGE_MIN_EDGE} and {IMAGE_MAX_EDGE}."
        )
    if value % divisor != 0:
        raise ImageGenerateError(f"{name} must be divisible by {divisor}.")


def image_generate_options(
    constraints: dict[str, Any] | None,
    prompt: str,
    *,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Build the native image fields this model will accept.

    Width and height are included only for models that size by pixels
    (a width/height divisor, and no aspect-ratio or resolution list).
    Aspect ratio and resolution are included only when that list exists
    and the value is in it. Anything else raises ImageGenerateError so
    Venice is not asked for a size it will reject with HTTP 400.
    """
    spec = constraints if isinstance(constraints, dict) else {}
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageGenerateError("Prompt is required.")
    limit = _positive_int(spec.get("promptCharacterLimit"))
    if limit is not None and len(prompt) > limit:
        raise ImageGenerateError(
            f"Prompt is {len(prompt)} characters; this model allows {limit}."
        )

    aspects = _constraint_choices(spec.get("aspectRatios"))
    resolutions = _constraint_choices(spec.get("resolutions"))
    divisor = _positive_int(spec.get("widthHeightDivisor"))
    uses_pixels = divisor is not None and not aspects and not resolutions

    options: dict[str, Any] = {
        "prompt": prompt,
        "format": IMAGE_GENERATE_FORMAT,
    }
    if aspect_ratio is not None:
        if not aspects:
            raise ImageGenerateError("This model does not accept aspect_ratio.")
        if aspect_ratio not in aspects:
            joined = ", ".join(aspects)
            raise ImageGenerateError(f"aspect_ratio must be one of: {joined}.")
        options["aspect_ratio"] = aspect_ratio
    if resolution is not None:
        if not resolutions:
            raise ImageGenerateError("This model does not accept resolution.")
        if resolution not in resolutions:
            joined = ", ".join(resolutions)
            raise ImageGenerateError(f"resolution must be one of: {joined}.")
        options["resolution"] = resolution
    if width is None and height is None:
        return options
    if not uses_pixels:
        raise ImageGenerateError("This model does not accept width or height.")
    if width is None or height is None:
        raise ImageGenerateError("width and height must be set together.")
    assert divisor is not None
    _check_edge("width", width, divisor)
    _check_edge("height", height, divisor)
    options["width"] = width
    options["height"] = height
    return options


class Images:
    """Native image generation for Venice AI."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize images API."""
        self.client = client

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        constraints: dict[str, Any] | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Generate an image with POST /image/generate.

        ``constraints`` is that model's ``model_spec.constraints``. Sizing
        fields the model does not list are rejected here and are not sent.
        ``"default"`` is not a model id on this endpoint.
        """
        if not isinstance(model, str) or not model.strip() or model == "default":
            raise ImageGenerateError(
                'model must be a Venice image model id, not "default".'
            )
        options = image_generate_options(
            constraints,
            prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            width=width,
            height=height,
        )
        payload = {"model": model, **options}
        self.client.metrics.record_request()

        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/image/generate",
                headers=self.client._headers,
                json=payload,
                timeout=DEFAULT_IMAGE_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as err:
            error_detail = getattr(err.response, "text", str(err))
            _LOGGER.error(
                "Venice AI Images API HTTP error %s: %s",
                err.response.status_code,
                error_detail,
            )
            categorized = _categorize_http_error(
                err.response.status_code, error_detail, "generating image"
            )
            self.client.metrics.record_error(categorized)
            raise categorized from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Images API request error: %s", err)
            network_err = NetworkError(f"Request error generating image: {err}")
            self.client.metrics.record_error(network_err)
            raise network_err from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode images JSON response: %s", response.text)
            decode_err = VeniceAIError("Failed to decode images API response")
            self.client.metrics.record_error(decode_err)
            raise decode_err from err



class AsyncVeniceAIClient:
    """Async client for the Venice AI API using httpx."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.venice.ai/api/v1",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the client."""
        # SEC-1: store the credential unmodified so any diagnostics or
        # re-auth round-trip sees the user-provided value. The CR/LF-only
        # scrub is applied at header-construction time below. (Previous
        # code mutated the key via .strip(), which turned valid keys
        # into 401-rejected keys — regression in commit 64b115c.)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # QUAL-2 / PERF-4: pool sizing and default timeout sourced from constants
        # so a single edit in const.py changes the whole client.
        self._http_client = http_client if http_client else httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_HTTP_TIMEOUT),
            limits=httpx.Limits(
                max_keepalive_connections=DEFAULT_HTTP_KEEPALIVE,
                max_connections=DEFAULT_HTTP_MAX_CONNECTIONS,
            ),
        )
        self._should_close_client = not http_client
        self._closed = False

        # In-memory usage/telemetry counters backing the diagnostic sensor
        # entities (LOW-4). Shared across all sub-API helpers via ``self``.
        self.metrics = VeniceAIMetrics()

        self._headers = {
            "Authorization": f"Bearer {_sanitize_header_value(api_key)}",
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, br",
        }
        self.chat = ChatCompletions(self)
        self.models = Models(self)
        self.speech = Speech(self)
        self.transcriptions = Transcriptions(self)
        self.images = Images(self)

    def authorization_header(self) -> str:
        """Return the CR/LF-scrubbed bearer value used on every endpoint."""
        return self._headers["Authorization"]

    async def _async_request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff retry for transient failures.

        Retry configuration is sourced from the module-level constants
        MAX_RETRIES, RETRY_BASE_DELAY, and RETRY_MAX_DELAY (defined in
        const.py — MED-4) so they can be tuned without touching this method.
        """
        retryable_statuses = {429, 500, 502, 503}
        url = f"{self._base_url}{endpoint}"
        _req_start = time.monotonic()

        for attempt in range(MAX_RETRIES + 1):
            _attempt_start = time.monotonic()
            if attempt == 0:
                _LOGGER.debug("[PERF-HTTP] %s %s — sending request", method, endpoint)
            else:
                _LOGGER.debug(
                    "[PERF-HTTP] %s %s — retry attempt %d/%d",
                    method, endpoint, attempt + 1, MAX_RETRIES + 1,
                )
            try:
                response = await self._http_client.request(method, url, **kwargs)
                _attempt_elapsed = time.monotonic() - _attempt_start

                if response.status_code in retryable_statuses:
                    if attempt < MAX_RETRIES:
                        # Read and close before the next attempt. aread() alone
                        # does not release the connection if the body read fails.
                        try:
                            await response.aread()
                        except Exception:
                            _LOGGER.debug(
                                "Could not read retry response body for %s %s",
                                method,
                                endpoint,
                                exc_info=True,
                            )
                        try:
                            await response.aclose()
                        except Exception:
                            _LOGGER.debug(
                                "Could not close retry response for %s %s",
                                method,
                                endpoint,
                                exc_info=True,
                            )
                        delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                        _LOGGER.warning(
                            "[PERF-HTTP] %s %s → HTTP %d in %.3fs; retrying in %.1fs (attempt %d/%d)",
                            method, endpoint, response.status_code, _attempt_elapsed,
                            delay, attempt + 1, MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue

                _LOGGER.debug(
                    "[PERF-HTTP] %s %s → HTTP %d in %.3fs%s",
                    method, endpoint, response.status_code, _attempt_elapsed,
                    f" ({attempt} retr{'y' if attempt == 1 else 'ies'}, {time.monotonic() - _req_start:.3f}s total)"
                    if attempt > 0 else "",
                )
                return response

            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as err:
                _attempt_elapsed = time.monotonic() - _attempt_start
                if attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    _LOGGER.warning(
                        "[PERF-HTTP] %s %s → %s after %.3fs; retrying in %.1fs (attempt %d/%d)",
                        method, endpoint, type(err).__name__, _attempt_elapsed,
                        delay, attempt + 1, MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                else:
                    _LOGGER.warning(
                        "[PERF-HTTP] %s %s → %s after %.3fs; max retries exhausted (%.3fs total)",
                        method, endpoint, type(err).__name__, _attempt_elapsed,
                        time.monotonic() - _req_start,
                    )
                    raise NetworkError(f"Max retries exceeded: {err}") from err

        # Should never reach here; all retry attempts exhausted
        raise NetworkError("Max retries exceeded")

    async def close(self) -> None:
        """Close the httpx client if it was created internally."""
        if self._closed:
            return
        self._closed = True
        if self._should_close_client:
            await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
