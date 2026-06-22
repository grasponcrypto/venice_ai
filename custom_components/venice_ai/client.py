"""Venice AI API Client."""
from __future__ import annotations

import asyncio
import json
import logging
import time
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
# CACHE_TTL_SECONDS. The cache stores plain dicts so it is independent of any
# specific httpx client lifetime.
_PROCESS_MODEL_CACHE: dict[str, tuple[float, list[dict]]] = {}
_PROCESS_MODEL_CACHE_TTL = 3600  # seconds; see also Models._CACHE_TTL_SECONDS


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

    def record_request(self) -> None:
        """Increment the total request counter."""
        self.request_count += 1

    def record_error(self, error: BaseException) -> None:
        """Increment the error counter and remember the last error message."""
        self.error_count += 1
        self.last_error = f"{type(error).__name__}: {error}"

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """Accumulate token usage from an API ``usage`` block, if present."""
        if not isinstance(usage, dict):
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)



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
    ) -> AsyncGenerator[AsyncGenerator[ChatCompletionChunk, None], None]:
        """Create a streaming chat completion."""
        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if venice_parameters is not None:
            data["venice_parameters"] = venice_parameters
        if tools is not None:
            data["tools"] = tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice

        response: httpx.Response | None = None
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

        except httpx.HTTPStatusError as err:
            if response is not None:
                await response.aclose()
            error_detail = ""
            try:
                error_detail = err.response.text
            except Exception:
                pass
            _LOGGER.error("Venice AI HTTP error %s: %s", err.response.status_code, error_detail)
            raise _categorize_http_error(err.response.status_code, error_detail, "streaming chat") from err
        except httpx.RequestError as err:
            if response is not None:
                await response.aclose()
            _LOGGER.error("Venice AI request error: %s", err)
            raise NetworkError(f"Request error (streaming chat): {err}") from err

        async def _stream() -> AsyncGenerator[ChatCompletionChunk, None]:
            try:
                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk_data = json.loads(line[6:])
                            yield ChatCompletionChunk(chunk_data)
                        except json.JSONDecodeError:
                            _LOGGER.warning("Failed to decode stream chunk: %s", line)
                    else:
                        _LOGGER.warning("Received unexpected line in stream: %s", line)
            except httpx.TransportError as err:
                # Convert mid-stream transport failures (connection reset, server close,
                # timeout) to a typed NetworkError so callers get consistent exceptions.
                _LOGGER.error("Stream interrupted by transport error: %s", err)
                raise NetworkError(f"Stream interrupted: {err}") from err
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
        payload = {**payload, "stream": False}

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
            # Try to extract a human-readable message from the JSON error body.
            error_message = error_detail
            if error_detail:
                try:
                    error_json = json.loads(error_detail)
                    if isinstance(error_json.get("error"), dict):
                        error_message = error_json["error"].get("message", error_detail)
                    elif isinstance(error_json.get("error"), str):
                        error_message = error_json["error"]
                except json.JSONDecodeError:
                    pass
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

    _CACHE_TTL_SECONDS = 3600  # 1 hour

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize models API."""
        self.client = client
        self._cache: dict[str, tuple[list[dict], float]] = {}

    async def list(self, model_type: str = "text") -> list[dict]:
        """List available models with TTL caching.

        PERF-1: consults a process-wide cache first so multiple client
        instances (config entries, tests) share one network round-trip per
        model_type per TTL window. Falls back to the per-instance cache
        (``self._cache``) and then to the live API if both are cold.
        """
        now = time.monotonic()
        # PERF-1: process-wide cache check (keyed by model_type)
        global_cached = _PROCESS_MODEL_CACHE.get(model_type)
        if global_cached is not None:
            timestamp, models = global_cached
            if now - timestamp < _PROCESS_MODEL_CACHE_TTL:
                _LOGGER.debug(
                    "PERF-1: process-wide cache hit for %s models (%d entries)",
                    model_type,
                    len(models),
                )
                return models
        cached = self._cache.get(model_type)
        if cached is not None:
            models, timestamp = cached
            if now - timestamp < self._CACHE_TTL_SECONDS:
                _LOGGER.debug("Returning cached %s models (%d entries, age=%.0fs)", model_type, len(models), now - timestamp)
                return models
            _LOGGER.debug("Cache expired for %s models, fetching fresh", model_type)

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
            self._cache[model_type] = (models, now)
            # PERF-1: write through to process-wide cache so other clients in
            # the same Python process can reuse this result.
            _PROCESS_MODEL_CACHE[model_type] = (now, models)
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
        data = {
            "input": text,
            "model": model,
            "voice": voice,
            "response_format": audio_output,
            "speed": speed,
        }

        audio_headers = {
            "Authorization": f"Bearer {self.client._api_key}",
        }

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
            _LOGGER.debug("Received raw audio data: %d bytes", len(audio_data))
            return audio_data

        except httpx.HTTPStatusError as err:
            error_detail = "Unknown error"
            try:
                if err.response.headers.get("content-type", "").startswith("audio/"):
                    error_detail = f"HTTP {err.response.status_code} for audio request"
                else:
                    error_detail = err.response.text[:500]
            except Exception:
                error_detail = f"HTTP {err.response.status_code}"

            _LOGGER.error("Venice AI Speech API HTTP error %s: %s", err.response.status_code, error_detail)
            raise _categorize_http_error(err.response.status_code, error_detail, "generating speech") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Speech API request error: %s", err)
            raise NetworkError(f"Request error generating speech: {err}") from err

    async def generate_streaming(
        self,
        text: str,
        voice: str = "bm_daniel",
        model: str = "tts-kokoro",
        audio_output: str = "mp3",
        speed: float = 1.0,
    ) -> AsyncGenerator[bytes, None]:
        """Generate speech audio from text with streaming chunks."""
        data = {
            "input": text,
            "model": model,
            "voice": voice,
            "response_format": audio_output,
            "speed": speed,
            "streaming": True,
        }

        audio_headers = {
            "Authorization": f"Bearer {self.client._api_key}",
        }

        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/audio/speech",
                headers=audio_headers,
                json=data,
                timeout=DEFAULT_TTS_TIMEOUT,
            )
            response.raise_for_status()

            _LOGGER.debug("Streaming TTS response received, yielding chunks")
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            except httpx.TransportError as err:
                _LOGGER.error("Streaming TTS transport error: %s", err)
                raise NetworkError(f"Streaming TTS interrupted: {err}") from err

        except httpx.HTTPStatusError as err:
            error_detail = "Unknown error"
            try:
                if err.response.headers.get("content-type", "").startswith("audio/"):
                    error_detail = f"HTTP {err.response.status_code} for audio request"
                else:
                    error_detail = err.response.text[:500]
            except Exception:
                error_detail = f"HTTP {err.response.status_code}"

            _LOGGER.error("Venice AI Speech API HTTP error %s: %s", err.response.status_code, error_detail)
            raise _categorize_http_error(err.response.status_code, error_detail, "streaming speech") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Speech API request error: %s", err)
            raise NetworkError(f"Request error generating streaming speech: {err}") from err


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
            "Authorization": f"Bearer {self.client._api_key}",
        }

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
            if response_format == "json":
                return response.json()
            else:
                return {"text": response.text}

        except httpx.HTTPStatusError as err:
            error_detail = getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI Transcriptions API HTTP error %s: %s", err.response.status_code, error_detail)
            raise _categorize_http_error(err.response.status_code, error_detail, "creating transcription") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Transcriptions API request error: %s", err)
            raise NetworkError(f"Request error creating transcription: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode transcriptions JSON response: %s", response.text)
            raise VeniceAIError("Failed to decode transcriptions API response") from err


class Images:
    """Images API for Venice AI."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize images API."""
        self.client = client

    async def generate(
        self,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        response_format: str = "url",
        n: int = 1,
    ) -> dict[str, Any]:
        """Generate an image with Venice AI."""
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "style": style,
            "response_format": response_format,
            "n": n,
        }

        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/images/generations",
                headers=self.client._headers,
                json=payload,
                timeout=DEFAULT_IMAGE_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as err:
            error_detail = getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI Images API HTTP error %s: %s", err.response.status_code, error_detail)
            raise _categorize_http_error(err.response.status_code, error_detail, "generating image") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Images API request error: %s", err)
            raise NetworkError(f"Request error generating image: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode images JSON response: %s", response.text)
            raise VeniceAIError("Failed to decode images API response") from err


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

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._http_client.request(method, url, **kwargs)

                if response.status_code in retryable_statuses:
                    if attempt < MAX_RETRIES:
                        # Fully consume response body to free connection before retry
                        try:
                            await response.aread()
                        except Exception:
                            pass
                        delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                        _LOGGER.warning(
                            "Venice AI API returned HTTP %d, retrying in %.1fs (attempt %d/%d)",
                            response.status_code, delay, attempt + 1, MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue

                return response

            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as err:
                if attempt < MAX_RETRIES:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    _LOGGER.warning(
                        "Venice AI API request error (%s), retrying in %.1fs (attempt %d/%d)",
                        type(err).__name__, delay, attempt + 1, MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                else:
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
