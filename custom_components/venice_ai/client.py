"""Venice AI API Client."""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, cast
import time

import httpx

class VeniceAIError(Exception):
    """Base exception for Venice AI errors."""

class AuthenticationError(VeniceAIError):
    """Authentication error."""

class ChatCompletionChunk:
    """Chat completion chunk."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize chat completion chunk."""
        self.choices = data.get("choices", [])

class ChatCompletions:
    """Chat completions API."""

    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize chat completions."""
        self.client = client

    async def create(
        self,
        model: str,
        messages: list[dict[str, Any]],  # Updated to Any for tool_calls
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        venice_parameters: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Create a chat completion."""
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
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

        try:
            async with self.client._http_client.stream(
                "POST",
                f"{self.client._base_url}/chat/completions",
                headers=self.client._headers,
                json=data,
                timeout=300.0
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip() and line != "data: [DONE]":
                        yield ChatCompletionChunk(json.loads(line[6:]))
                        
        except httpx.HTTPStatusError as err:
            if err.response.status_code == 401:
                raise AuthenticationError("Invalid API key") from err
            raise VeniceAIError(f"HTTP error {err.response.status_code}") from err

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
        self._http_client = http_client or httpx.AsyncClient()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.chat = ChatCompletions(self)
        self.models = Models(self)
        self._models_cache = None
        self._models_cache_time = 0

    async def close(self) -> None:
        """Close the client."""
        await self._http_client.aclose()

class Models:
    """Models API for Venice AI."""
    
    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize models API."""
        self.client = client

    async def list(self) -> list[dict]:
        """List available models."""
        response = await self.client._http_client.get(
            f"{self.client._base_url}/models",
            headers=self.client._headers,
            params={"type": "text"}
        )
        response.raise_for_status()
        return response.json().get("data", [])