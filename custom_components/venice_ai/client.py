"""Venice AI API Client."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator

import httpx

_LOGGER = logging.getLogger(__name__)


class VeniceAIError(Exception):
    """Base exception for Venice AI errors."""


class AuthenticationError(VeniceAIError):
    """Authentication error."""


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
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
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
                timeout=300.0,
            )
            response = await self.client._http_client.send(request, stream=True)
            response.raise_for_status()

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

        except httpx.HTTPStatusError as err:
            error_detail = err.response.text
            _LOGGER.error("Venice AI HTTP error %s: %s", err.response.status_code, error_detail)
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key") from err
            raise VeniceAIError(f"HTTP error {err.response.status_code}: {error_detail}") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI request error: %s", err)
            raise VeniceAIError(f"Request error: {err}") from err
        finally:
            if response is not None:
                await response.aclose()

    async def create_non_streaming(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a non-streaming chat completion."""
        payload["stream"] = False
        response_text = None

        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/chat/completions",
                headers=self.client._headers,
                json=payload,
                timeout=120.0,
            )
            response_text = response.text
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as err:
            error_detail = response_text if response_text is not None else getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI HTTP error %s: %s", err.response.status_code, error_detail)
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key") from err
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
            raise VeniceAIError(f"HTTP error {err.response.status_code}: {error_message}") from err

        except httpx.RequestError as err:
            _LOGGER.error("Venice AI request error: %s", err)
            raise VeniceAIError(f"Request error: {err}") from err

        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode non-streaming JSON response: %s", response_text)
            raise VeniceAIError(f"Failed to decode API response: {response_text}") from err


class Models:
    """Models API for Venice AI with TTL caching."""

    _CACHE_TTL_SECONDS = 3600  # 1 hour

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize models API."""
        self.client = client
        self._cache: dict[str, tuple[list[dict], float]] = {}

    async def list(self, model_type: str = "text") -> list[dict]:
        """List available models with TTL caching."""
        now = time.monotonic()
        cached = self._cache.get(model_type)
        if cached is not None:
            models, timestamp = cached
            if now - timestamp < self._CACHE_TTL_SECONDS:
                _LOGGER.debug("Returning cached %s models (%d entries, age=%.0fs)", model_type, len(models), now - timestamp)
                return models
            _LOGGER.debug("Cache expired for %s models, fetching fresh", model_type)

        response_text = None
        url = f"{self.client._base_url}/models"
        _LOGGER.debug("Attempting to fetch %s models from URL: %s", model_type, url)
        try:
            response = await self.client._async_request_with_retry(
                "GET",
                "/models",
                headers=self.client._headers,
                params={"type": model_type},
            )
            response_text = response.text
            response.raise_for_status()
            model_data = response.json()
            models = model_data.get("data", [])
            self._cache[model_type] = (models, now)
            _LOGGER.debug("Successfully fetched %d %s models", len(models), model_type)
            return models
        except httpx.HTTPStatusError as err:
            error_detail = response_text if response_text is not None else getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI Models API HTTP error %s: %s", err.response.status_code, error_detail)
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key checking models") from err
            raise VeniceAIError(f"HTTP error fetching models {err.response.status_code}: {error_detail}") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Models API request error: %s (URL: %s, type: %s)", err, url, type(err).__name__)
            raise VeniceAIError(f"Request error fetching models: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode models JSON response: %s", response_text)
            raise VeniceAIError("Failed to decode models API response") from err


class Voices:
    """Voices API for Venice AI."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize voices API."""
        self.client = client

    async def list(self) -> list[dict]:
        """List available voices."""
        response_text = None
        try:
            response = await self.client._async_request_with_retry(
                "GET",
                "/audio/voices",
                headers=self.client._headers,
            )
            response_text = response.text
            response.raise_for_status()
            voice_data = response.json()
            voices = voice_data.get("data", [])
            return voices
        except httpx.HTTPStatusError as err:
            error_detail = response_text if response_text is not None else getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI Voices API HTTP error %s: %s", err.response.status_code, error_detail)
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key for voices") from err
            raise VeniceAIError(f"HTTP error fetching voices {err.response.status_code}: {error_detail}") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Voices API request error: %s", err)
            raise VeniceAIError(f"Request error fetching voices: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode voices JSON response: %s", response_text)
            raise VeniceAIError("Failed to decode voices API response") from err


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
        streaming: bool = False,
    ) -> bytes:
        """Generate speech audio from text."""
        data = {
            "input": text,
            "model": model,
            "voice": voice,
            "response_format": audio_output,
            "speed": speed,
            "streaming": streaming,
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
                timeout=60.0,
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
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key for speech") from err
            raise VeniceAIError(f"HTTP error generating speech {err.response.status_code}: {error_detail}") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Speech API request error: %s", err)
            raise VeniceAIError(f"Request error generating speech: {err}") from err


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

        response_text = None
        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/audio/transcriptions",
                headers=multipart_headers,
                files=files,
                data=data,
                timeout=60.0,
            )
            response_text = response.text
            response.raise_for_status()
            if response_format == "json":
                return response.json()
            else:
                return {"text": response_text}

        except httpx.HTTPStatusError as err:
            error_detail = response_text if response_text is not None else getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI Transcriptions API HTTP error %s: %s", err.response.status_code, error_detail)
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key for transcriptions") from err
            raise VeniceAIError(f"HTTP error creating transcription {err.response.status_code}: {error_detail}") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Transcriptions API request error: %s", err)
            raise VeniceAIError(f"Request error creating transcription: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode transcriptions JSON response: %s", response_text)
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

        response_text = None
        try:
            response = await self.client._async_request_with_retry(
                "POST",
                "/images/generations",
                headers=self.client._headers,
                json=payload,
                timeout=120.0,
            )
            response_text = response.text
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as err:
            error_detail = response_text if response_text is not None else getattr(err.response, "text", str(err))
            _LOGGER.error("Venice AI Images API HTTP error %s: %s", err.response.status_code, error_detail)
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key for image generation") from err
            raise VeniceAIError(f"HTTP error generating image {err.response.status_code}: {error_detail}") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI Images API request error: %s", err)
            raise VeniceAIError(f"Request error generating image: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode images JSON response: %s", response_text)
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
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client if http_client else httpx.AsyncClient()
        self._should_close_client = not http_client

        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, br",
        }
        self.chat = ChatCompletions(self)
        self.models = Models(self)
        self.voices = Voices(self)
        self.speech = Speech(self)
        self.transcriptions = Transcriptions(self)
        self.images = Images(self)

    async def _async_request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff retry for transient failures."""
        max_retries = 3
        retryable_statuses = {429, 500, 502, 503}
        base_delay = 1.0
        last_err = None

        url = f"{self._base_url}{endpoint}"

        for attempt in range(max_retries + 1):
            try:
                response = await self._http_client.request(method, url, **kwargs)

                if response.status_code in retryable_statuses:
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), 30.0)
                        _LOGGER.warning(
                            "Venice AI API returned HTTP %d, retrying in %.1fs (attempt %d/%d)",
                            response.status_code, delay, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue

                return response

            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as err:
                last_err = err
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), 30.0)
                    _LOGGER.warning(
                        "Venice AI API request error (%s), retrying in %.1fs (attempt %d/%d)",
                        type(err).__name__, delay, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise last_err from err

        raise VeniceAIError("Max retries exceeded")

    async def close(self) -> None:
        """Close the httpx client if it was created internally."""
        if self._should_close_client:
            await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
