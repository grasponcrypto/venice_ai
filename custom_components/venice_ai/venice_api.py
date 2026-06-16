"""Service layer for Venice AI (ARCH-1 / ARCH-2).

This module separates *domain logic* from the Home Assistant platform code.
Platform entities (conversation, TTS, STT, AI Task) should delegate their
API interaction to the service classes defined here, keeping the platform
files a thin integration layer.

Design goals (ARCH-1, ARCH-2):
- A single place that encapsulates Venice AI API interaction patterns.
- Reusable, independently testable business logic (see ``tests/``).
- Consistent parameter handling and error surfacing across platforms.

The :class:`VeniceConversationService` additionally implements streaming chat
support (MED-3) via :meth:`VeniceConversationService.chat_stream`, accumulating
streamed deltas (including tool-call fragments) into a final assistant message
that mirrors the non-streaming response shape so callers can use either mode
interchangeably.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from .client import AsyncVeniceAIClient

_LOGGER = logging.getLogger(__name__)


@dataclass
class ChatParameters:
    """Tunable parameters for a chat completion request.

    Centralises the per-request knobs so platform code can build one object
    from config options instead of threading many positional arguments through
    the service methods.
    """

    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    venice_parameters: dict[str, Any] | None = None


@dataclass
class StreamingChatResult:
    """Accumulated result of a streamed chat completion.

    Mirrors the relevant parts of a non-streaming response so callers can
    handle streaming and non-streaming paths with the same downstream logic.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def as_message(self) -> dict[str, Any]:
        """Return an assistant ``message`` dict like the non-streaming API."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message

    def as_response(self) -> dict[str, Any]:
        """Return a response envelope shaped like the non-streaming API."""
        return {"choices": [{"message": self.as_message()}]}


class VeniceConversationService:
    """Encapsulate chat-completion interaction patterns for conversations.

    This service owns the *how* of talking to the Venice AI chat API (request
    construction, streaming accumulation, response shaping) while the
    conversation entity owns the *what* (chat-log management, tool dispatch,
    Home Assistant intent responses).
    """

    def __init__(self, client: AsyncVeniceAIClient) -> None:
        """Initialize the service with a Venice AI client."""
        self._client = client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        params: ChatParameters,
    ) -> dict[str, Any]:
        """Perform a non-streaming chat completion and return the raw response."""
        return await self._client.chat.create_non_streaming(
            model=params.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            tools=params.tools or None,
            venice_parameters=params.venice_parameters,
            stream=False,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        params: ChatParameters,
        on_delta: Any | None = None,
    ) -> StreamingChatResult:
        """Perform a streaming chat completion (MED-3).

        Streams chat-completion chunks from the Venice AI API and accumulates
        them into a :class:`StreamingChatResult`. Content deltas are appended in
        order; tool-call fragments are merged by their ``index`` so partial
        function-name/argument strings emitted across chunks are reassembled
        into complete tool calls that match the non-streaming response shape.

        Args:
            messages: The conversation messages to send.
            params: Chat parameters (model, sampling, tools, etc.).
            on_delta: Optional callable invoked with each text delta as it
                arrives, enabling incremental UI updates. May be a coroutine
                function or a plain callable.

        Returns:
            A :class:`StreamingChatResult` with the fully accumulated content
            and any reconstructed tool calls.
        """
        result = StreamingChatResult()
        # Tool calls keyed by their streaming ``index`` for incremental merge.
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        async with self._client.chat.create(
            model=params.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            tools=params.tools or None,
            venice_parameters=params.venice_parameters,
        ) as stream:
            async for chunk in stream:
                for choice in chunk.choices:
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                    if not isinstance(delta, dict):
                        continue

                    content_piece = delta.get("content")
                    if content_piece:
                        result.content += content_piece
                        if on_delta is not None:
                            await _maybe_await(on_delta, content_piece)

                    for tc in delta.get("tool_calls", []) or []:
                        _merge_tool_call_fragment(tool_calls_by_index, tc)

        # Emit reconstructed tool calls in index order.
        result.tool_calls = [
            tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
        ]
        _LOGGER.debug(
            "Streaming chat complete: %d chars, %d tool call(s)",
            len(result.content),
            len(result.tool_calls),
        )
        return result


def _merge_tool_call_fragment(
    accumulator: dict[int, dict[str, Any]],
    fragment: dict[str, Any],
) -> None:
    """Merge a streamed tool-call delta fragment into the accumulator.

    Streaming APIs send tool calls in pieces: the first fragment for a given
    ``index`` typically carries the ``id`` and function ``name`` while later
    fragments append to the function ``arguments`` string. This reassembles
    them into a single OpenAI-style tool-call dict.
    """
    if not isinstance(fragment, dict):
        return
    index = fragment.get("index", 0)
    existing = accumulator.setdefault(
        index,
        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
    )

    if fragment.get("id"):
        existing["id"] = fragment["id"]
    if fragment.get("type"):
        existing["type"] = fragment["type"]

    func = fragment.get("function", {})
    if isinstance(func, dict):
        if func.get("name"):
            existing["function"]["name"] += func["name"]
        if func.get("arguments"):
            existing["function"]["arguments"] += func["arguments"]


async def _maybe_await(callback: Any, *args: Any) -> None:
    """Invoke ``callback`` whether it is sync or async."""
    import asyncio

    result = callback(*args)
    if asyncio.iscoroutine(result):
        await result
