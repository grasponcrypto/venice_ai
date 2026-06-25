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
from typing import Any

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
class ToolCall:
    """PERF-3: strongly-typed wrapper around a single tool call.

    Streamed or non-streamed tool calls from the Venice AI chat API are raw
    ``dict`` objects with arbitrary string-typed arguments. Wrapping them in
    a dataclass gives downstream consumers type-checked access to the id,
    function name, and arguments, and lets the conversation entity cache
    the parsed argument ``dict`` to avoid re-parsing the same JSON string
    on every iteration.

    ``args_dict`` is lazily populated by :meth:`parsed_args`; the raw
    ``arguments`` string is preserved for fidelity with the API payload.
    """

    id: str
    call_type: str
    function_name: str
    arguments: str
    args_dict: dict[str, Any] | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ToolCall | None":
        """Return a :class:`ToolCall` from a raw tool-call dict, or None on garbage input.

        Performs minimal validation: ``id`` and function ``name`` must be
        non-empty strings. Returns ``None`` for malformed payloads so the
        caller can decide whether to log + skip or propagate an error.
        """
        if not isinstance(raw, dict):
            return None
        func = raw.get("function")
        if not isinstance(func, dict):
            return None
        call_id = raw.get("id")
        name = func.get("name")
        if not isinstance(call_id, str) or not call_id:
            return None
        if not isinstance(name, str) or not name:
            return None
        return cls(
            id=call_id,
            call_type=str(raw.get("type", "function")),
            function_name=name,
            arguments=str(func.get("arguments", "")),
        )

    def parsed_args(self) -> dict[str, Any]:
        """Return the decoded arguments dict, caching the result on first parse."""
        if self.args_dict is not None:
            return self.args_dict
        import json

        try:
            decoded = json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            decoded = {}
        if not isinstance(decoded, dict):
            # SEC-2 (defensive): if the args somehow round-trip to a
            # non-object, normalise to an empty dict. Callers should
            # already have rejected non-dict args before invocation.
            decoded = {}
        self.args_dict = decoded
        return decoded


@dataclass
class StreamingChatResult:
    """Accumulated result of a streamed chat completion.

    Mirrors the relevant parts of a non-streaming response so callers can
    handle streaming and non-streaming paths with the same downstream logic.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "unknown"
    time_to_first_token: float | None = None  # seconds from stream open to first content delta
    usage: dict[str, Any] = field(default_factory=dict)  # token usage from final stream chunk

    def as_message(self) -> dict[str, Any]:
        """Return an assistant ``message`` dict like the non-streaming API."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message

    def as_response(self) -> dict[str, Any]:
        """Return a response envelope shaped like the non-streaming API."""
        resp: dict[str, Any] = {
            "choices": [{
                "message": self.as_message(),
                "finish_reason": self.finish_reason,
            }]
        }
        if self.usage:
            resp["usage"] = self.usage
        return resp

    def typed_tool_calls(self) -> list[ToolCall]:
        """PERF-3: return tool calls as :class:`ToolCall` objects, skipping malformed ones."""
        typed: list[ToolCall] = []
        for raw in self.tool_calls:
            tc = ToolCall.from_raw(raw)
            if tc is not None:
                typed.append(tc)
        return typed


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
        import time as _time
        result = StreamingChatResult()
        # Tool calls keyed by their streaming ``index`` for incremental merge.
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        _stream_open_t = _time.monotonic()
        _LOGGER.debug(
            "[PERF] chat_stream — starting stream (model=%s, messages=%d, tools=%s)",
            params.model,
            len(messages),
            len(params.tools) if params.tools else 0,
        )

        async with self._client.chat.create(
            model=params.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            tools=params.tools or None,
            venice_parameters=params.venice_parameters,
            stream_options={"include_usage": True},
        ) as stream:
            async for chunk in stream:
                # Capture token usage emitted in the final chunk
                # (requires stream_options.include_usage=True on the request).
                if chunk.usage:
                    result.usage = chunk.usage
                    _LOGGER.debug(
                        "[PERF] chat_stream — usage chunk received: prompt=%s completion=%s total=%s",
                        chunk.usage.get("prompt_tokens"),
                        chunk.usage.get("completion_tokens"),
                        chunk.usage.get("total_tokens"),
                    )
                for choice in chunk.choices:
                    # The SDK may return choices as objects (with attributes) or
                    # as dicts depending on the SDK version. Support both.
                    if isinstance(choice, dict):
                        raw_delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")
                    else:
                        raw_delta = getattr(choice, "delta", None)
                        finish_reason = getattr(choice, "finish_reason", None)

                    if finish_reason:
                        result.finish_reason = finish_reason

                    # Normalise delta to a plain dict for uniform access.
                    if raw_delta is None:
                        continue
                    if isinstance(raw_delta, dict):
                        delta = raw_delta
                    else:
                        # Object-style delta: convert to dict via __dict__ or
                        # attribute access. getattr fallback prevents AttributeError
                        # on SDK-version mismatches.
                        delta = getattr(raw_delta, "__dict__", None)
                        if delta is None:
                            try:
                                delta = {
                                    "content": getattr(raw_delta, "content", None),
                                    "tool_calls": getattr(raw_delta, "tool_calls", None),
                                }
                            except Exception:
                                continue

                    content_piece = delta.get("content") if isinstance(delta, dict) else None
                    if content_piece:
                        if result.time_to_first_token is None:
                            result.time_to_first_token = _time.monotonic() - _stream_open_t
                        result.content += content_piece
                        if on_delta is not None:
                            await _maybe_await(on_delta, content_piece)

                    tool_calls_raw = delta.get("tool_calls") if isinstance(delta, dict) else None
                    for tc in tool_calls_raw or []:
                        if isinstance(tc, dict):
                            _merge_tool_call_fragment(tool_calls_by_index, tc)
                        else:
                            # Object-style tool call fragment
                            try:
                                func = getattr(tc, "function", None)
                                tc_dict = {
                                    "index": getattr(tc, "index", 0),
                                    "id": getattr(tc, "id", None),
                                    "type": getattr(tc, "type", "function"),
                                    "function": {
                                        "name": getattr(func, "name", "") if func else "",
                                        "arguments": getattr(func, "arguments", "") if func else "",
                                    },
                                }
                                _merge_tool_call_fragment(tool_calls_by_index, tc_dict)
                            except Exception as exc:
                                _LOGGER.debug("Skipping malformed tool-call fragment: %s", exc)

        # Emit reconstructed tool calls in index order.
        result.tool_calls = [
            tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
        ]
        _total_elapsed = _time.monotonic() - _stream_open_t
        _LOGGER.debug(
            "[PERF] chat_stream — complete: %d chars, %d tool call(s), finish_reason=%r, "
            "time_to_first_token=%.3fs, total_elapsed=%.3fs",
            len(result.content),
            len(result.tool_calls),
            result.finish_reason,
            result.time_to_first_token or 0.0,
            _total_elapsed,
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
