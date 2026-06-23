"""Tests for the ``venice_api`` service layer (ARCH-1/ARCH-2, MED-3).

These are integration-style tests (TEST-2): they exercise the real service
logic against a mocked Venice AI client, verifying request construction,
streaming accumulation, and tool-call fragment reassembly.
"""

from __future__ import annotations

import pytest

from .conftest import load_component_module

venice_api = load_component_module("venice_api")

ChatParameters = venice_api.ChatParameters
StreamingChatResult = venice_api.StreamingChatResult
VeniceConversationService = venice_api.VeniceConversationService


class TestStreamingChatResult:
    """Tests for the streaming-result envelope shaping."""

    def test_as_message_without_tool_calls(self) -> None:
        result = StreamingChatResult(content="hello")
        assert result.as_message() == {"role": "assistant", "content": "hello"}

    def test_as_message_with_tool_calls(self) -> None:
        result = StreamingChatResult(content="", tool_calls=[{"id": "1"}])
        msg = result.as_message()
        assert msg["tool_calls"] == [{"id": "1"}]

    def test_as_response_matches_non_streaming_shape(self) -> None:
        result = StreamingChatResult(content="hi")
        resp = result.as_response()
        assert resp["choices"][0]["message"]["content"] == "hi"


class TestToolCallFragmentMerge:
    """Tests for incremental tool-call reassembly during streaming."""

    def test_fragments_merge_by_index(self) -> None:
        acc: dict[int, dict] = {}
        venice_api._merge_tool_call_fragment(
            acc,
            {"index": 0, "id": "call_1", "function": {"name": "get_", "arguments": '{"a"'}},
        )
        venice_api._merge_tool_call_fragment(
            acc,
            {"index": 0, "function": {"name": "weather", "arguments": ":1}"}},
        )
        assert acc[0]["id"] == "call_1"
        assert acc[0]["function"]["name"] == "get_weather"
        assert acc[0]["function"]["arguments"] == '{"a":1}'

    def test_non_dict_fragment_ignored(self) -> None:
        acc: dict[int, dict] = {}
        venice_api._merge_tool_call_fragment(acc, "nonsense")  # type: ignore[arg-type]
        assert acc == {}

    def test_multiple_indices_kept_separate(self) -> None:
        acc: dict[int, dict] = {}
        venice_api._merge_tool_call_fragment(
            acc, {"index": 0, "function": {"name": "a"}}
        )
        venice_api._merge_tool_call_fragment(
            acc, {"index": 1, "function": {"name": "b"}}
        )
        assert acc[0]["function"]["name"] == "a"
        assert acc[1]["function"]["name"] == "b"


@pytest.mark.asyncio
class TestVeniceConversationService:
    """End-to-end-ish tests of the service against a mocked client."""

    async def test_chat_forwards_parameters(self, make_client) -> None:
        client = make_client(
            non_streaming_response={
                "choices": [{"message": {"content": "answer"}}]
            }
        )
        service = VeniceConversationService(client)
        params = ChatParameters(
            model="venice-llm", max_tokens=128, temperature=0.5, top_p=0.9
        )

        resp = await service.chat([{"role": "user", "content": "hi"}], params)

        assert resp["choices"][0]["message"]["content"] == "answer"
        sent = client.chat.last_non_streaming_kwargs
        assert sent["model"] == "venice-llm"
        assert sent["max_tokens"] == 128
        assert sent["temperature"] == 0.5
        assert sent["top_p"] == 0.9
        assert sent["stream"] is False

    async def test_chat_stream_accumulates_content(self, make_client, chunk) -> None:
        chunks = [
            chunk({"content": "Hello"}),
            chunk({"content": ", "}),
            chunk({"content": "world"}),
        ]
        client = make_client(chunks=chunks)
        service = VeniceConversationService(client)

        result = await service.chat_stream(
            [{"role": "user", "content": "hi"}],
            ChatParameters(model="venice-llm"),
        )

        assert result.content == "Hello, world"
        assert result.tool_calls == []

    async def test_chat_stream_invokes_on_delta(self, make_client, chunk) -> None:
        deltas: list[str] = []
        client = make_client(chunks=[chunk({"content": "a"}), chunk({"content": "b"})])
        service = VeniceConversationService(client)

        async def collector(piece: str) -> None:
            deltas.append(piece)

        await service.chat_stream(
            [{"role": "user", "content": "x"}],
            ChatParameters(model="m"),
            on_delta=collector,
        )

        assert deltas == ["a", "b"]

    async def test_chat_stream_reassembles_tool_calls(self, make_client, chunk) -> None:
        chunks = [
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "get_", "arguments": '{"x"'},
                        }
                    ]
                }
            ),
            chunk(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"name": "time", "arguments": ":1}"}}
                    ]
                }
            ),
        ]
        client = make_client(chunks=chunks)
        service = VeniceConversationService(client)

        result = await service.chat_stream(
            [{"role": "user", "content": "time?"}],
            ChatParameters(model="m"),
        )

        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call["id"] == "call_1"
        assert call["function"]["name"] == "get_time"
        assert call["function"]["arguments"] == '{"x":1}'

    async def test_chat_stream_handles_sync_on_delta(self, make_client, chunk) -> None:
        seen: list[str] = []
        client = make_client(chunks=[chunk({"content": "z"})])
        service = VeniceConversationService(client)

        await service.chat_stream(
            [{"role": "user", "content": "x"}],
            ChatParameters(model="m"),
            on_delta=seen.append,
        )

        assert seen == ["z"]
