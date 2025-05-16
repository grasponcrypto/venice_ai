"""Venice AI API Client."""
from __future__ import annotations

import json
import logging # Added logging
from typing import Any, AsyncGenerator, cast

import httpx

# Use the standard logger name consistent with other modules
_LOGGER = logging.getLogger(__package__)

class VeniceAIError(Exception):
    """Base exception for Venice AI errors."""


class AuthenticationError(VeniceAIError):
    """Authentication error."""


class ChatCompletionChunk:
    """Chat completion chunk (used for streaming responses)."""
    # This class remains as is, used only by the streaming 'create' method.
    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize chat completion chunk."""
        self.choices = data.get("choices", [])
        # Add other fields from streaming chunk if needed (e.g., usage)


class ChatCompletions:
    """Chat completions API for Venice AI."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize chat completions."""
        self.client = client

    # --- Existing Streaming Method ---
    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]], # Allow Any for tool calls structure
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        venice_parameters: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None, # Added tools parameter
        tool_choice: str | dict[str, Any] | None = None, # Added tool_choice if needed
        # ... other parameters like frequency_penalty etc. if needed ...
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Create a streaming chat completion."""
        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True, # Explicitly set stream to True
        }
        # Add optional parameters if provided
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if venice_parameters is not None:
            data["venice_parameters"] = venice_parameters
        if tools is not None:
            data["tools"] = tools # Pass tools to API
        if tool_choice is not None:
             data["tool_choice"] = tool_choice # Pass tool_choice if provided


        _LOGGER.debug("Sending streaming request to Venice AI: %s", data)
        try:
            async with self.client._http_client.stream(
                "POST",
                f"{self.client._base_url}/chat/completions",
                headers=self.client._headers,
                json=data,
                # Increased timeout for potentially long streaming responses
                timeout=300.0
            ) as response:
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
            # Check response body for more specific error if possible
            error_detail = err.response.text
            _LOGGER.error("Venice AI HTTP error %s: %s", err.response.status_code, error_detail)
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key") from err
            # Add other specific error code handling if needed (e.g., 400, 429, 500)
            raise VeniceAIError(f"HTTP error {err.response.status_code}: {error_detail}") from err
        except httpx.RequestError as err:
            _LOGGER.error("Venice AI request error: %s", err)
            raise VeniceAIError(f"Request error: {err}") from err


    # --- Non-Streaming Method (with SyntaxError fix and logging) ---
    async def create_non_streaming(
        self,
        payload: dict[str, Any], # Accept the full payload dict directly
    ) -> dict[str, Any]:
        """Create a non-streaming chat completion."""
        # Ensure stream is explicitly false or absent
        payload["stream"] = False
        response_text = None # Initialize for potential use in error logging

        # Log the payload right before sending
        _LOGGER.debug("Sending non-streaming request payload to Venice AI: %s", json.dumps(payload, indent=2))
        try:
            response = await self.client._http_client.post(
                f"{self.client._base_url}/chat/completions",
                headers=self.client._headers,
                json=payload, # Send the actual dict, httpx handles serialization
                timeout=120.0 # Adjust timeout as needed
            )
            response_text = response.text # Store text before potential raise_for_status
            response.raise_for_status()
            # Attempt to parse JSON only if request was successful status-wise
            return response.json()

        except httpx.HTTPStatusError as err:
            # Use response_text if available, otherwise use err.response.text if available
            error_detail = response_text if response_text is not None else getattr(err.response, 'text', str(err))
            _LOGGER.error("Venice AI HTTP error %s: %s", err.response.status_code, error_detail)
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key") from err
            # Attempt to parse error details from response if JSON
            error_message = error_detail # Default message is the raw text
            if error_detail:
                try:
                     # Try parsing the stored text
                     error_json = json.loads(error_detail)
                     # Check common error structures
                     if isinstance(error_json.get('error'), dict):
                          error_message = error_json['error'].get('message', error_detail)
                     elif isinstance(error_json.get('error'), str):
                          error_message = error_json['error']
                except json.JSONDecodeError:
                     # Keep error_message as the raw text if JSON parsing fails
                     pass

            raise VeniceAIError(f"HTTP error {err.response.status_code}: {error_message}") from err

        except httpx.RequestError as err:
            _LOGGER.error("Venice AI request error: %s", err)
            raise VeniceAIError(f"Request error: {err}") from err

        except json.JSONDecodeError as err:
            # Log the response text that failed to parse
            _LOGGER.error("Failed to decode non-streaming JSON response: %s", response_text)
            raise VeniceAIError(f"Failed to decode API response: {response_text}") from err


class Models:
    """Models API for Venice AI."""

    def __init__(self, client: "AsyncVeniceAIClient") -> None:
        """Initialize models API."""
        self.client = client
        # Caching logic removed for simplicity, add back if needed
        # self._models_cache: list[dict] | None = None
        # self._models_cache_time: float = 0
        # self._cache_ttl: float = 3600 # Cache models for 1 hour, example

    async def list(self) -> list[dict]:
        """List available models."""
        _LOGGER.debug("Fetching model list from Venice AI API")
        response_text = None # Initialize for error logging
        try:
            response = await self.client._http_client.get(
                f"{self.client._base_url}/models",
                headers=self.client._headers,
                params={"type": "text"} # Spec shows type parameter
            )
            response_text = response.text
            response.raise_for_status()
            model_data = response.json()
            models = model_data.get("data", [])
            return models
        except httpx.HTTPStatusError as err:
             # Handle errors similarly to chat completion calls
             error_detail = response_text if response_text is not None else getattr(err.response, 'text', str(err))
             _LOGGER.error("Venice AI Models API HTTP error %s: %s", err.response.status_code, error_detail)
             if err.response.status_code == 401:
                 raise AuthenticationError("Invalid API key checking models") from err
             raise VeniceAIError(f"HTTP error fetching models {err.response.status_code}: {error_detail}") from err
        except httpx.RequestError as err:
             _LOGGER.error("Venice AI Models API request error: %s", err)
             raise VeniceAIError(f"Request error fetching models: {err}") from err
        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to decode models JSON response: %s", response_text)
            raise VeniceAIError("Failed to decode models API response") from err


# --- Main Async Client Class ---
class AsyncVeniceAIClient:
    """Async client for the Venice AI API using httpx."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.venice.ai/api/v1",
        # Allow passing httpx client for better HA integration
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the client."""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # Use provided httpx client or create a new one
        # Important: If creating new, it should be closed properly.
        # HA usually provides one via get_async_client.
        self._http_client = http_client if http_client else httpx.AsyncClient()
        self._should_close_client = not http_client # Flag if we created the client

        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Add Accept-Encoding if server supports compression for non-streaming
            "Accept-Encoding": "gzip, br",
        }
        # Initialize API endpoints
        self.chat = ChatCompletions(self)
        self.models = Models(self)
        # Note: Image generation client part is missing based on original file

    async def close(self) -> None:
        """Close the httpx client if it was created internally."""
        if self._should_close_client:
            await self._http_client.aclose()

    # Context manager methods for standalone use if needed
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()