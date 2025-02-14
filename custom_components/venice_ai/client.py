"""Venice AI API Client."""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, cast

import aiohttp

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
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        venice_parameters: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Create a chat completion."""
        data = {
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

        response = await self.client._http_client.post(
            f"{self.client._base_url}/chat/completions",
            headers=self.client._headers,
            json=data,
            timeout=300.0,
        )

        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        if response.status_code != 200:
            text = await response.aread()
            raise VeniceAIError(f"Error {response.status_code}: {text.decode()}")

        async for line in response.aiter_lines():
            line = line.strip()
            if not line or line == "data: [DONE]":
                continue
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            yield ChatCompletionChunk(data)

        await response.aclose()


class AsyncVeniceAIClient:
    """Async client for the Venice AI API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.venice.ai/v1",
        http_client: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize the client."""
        self._api_key = api_key
        self._base_url = base_url
        self._http_client = http_client or aiohttp.ClientSession()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.chat = ChatCompletions(self)

    async def close(self) -> None:
        """Close the client."""
        await self._http_client.close()

class Models:
    """Models API stub for compatibility."""
    
    async def list(self):
        """List available models."""
        return {"data": [{"id": "default"}]}

# Add models to the client
AsyncVeniceAIClient.models = Models() 