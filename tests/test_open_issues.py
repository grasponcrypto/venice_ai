"""Regression tests for the remaining 1.1.0 bug reports."""

from __future__ import annotations

import json

import pytest

from .conftest import load_component_module

catalog = load_component_module("model_catalog")
support = load_component_module("ai_task_support")
api = load_component_module("venice_api")
client = load_component_module("client")


def test_tts_catalog_is_not_judged_by_asr_models() -> None:
    audio = [{"id": "nvidia/parakeet-tdt-0.6b-v3", "model_type": "asr"}]
    data = {"audio_models": audio, "tts_catalog_ok": False, "asr_catalog_ok": True}
    assert catalog.model_ids(audio, "tts") == set()
    assert catalog.model_ids(audio, "asr") == {"nvidia/parakeet-tdt-0.6b-v3"}
    assert catalog.catalog_fetched(data, "tts_catalog_ok", audio, "tts") is False
    assert catalog.catalog_fetched(data, "asr_catalog_ok", audio, "asr") is True


def test_missing_catalog_flag_does_not_treat_an_empty_set_as_proof() -> None:
    audio = [{"id": "nvidia/parakeet-tdt-0.6b-v3", "model_type": "asr"}]
    assert catalog.catalog_fetched({}, "tts_catalog_ok", audio, "tts") is False
    assert catalog.catalog_fetched({}, "asr_catalog_ok", audio, "asr") is True


def test_malformed_tool_call_still_has_a_matching_error_result() -> None:
    missing_id = api.inspect_tool_call(
        {"type": "function", "function": {"name": "HassTurnOn"}}
    )
    assert missing_id.error == {"error": "Malformed tool call"}
    assert missing_id.call_id == ""

    bad_json = api.inspect_tool_call(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "HassTurnOn", "arguments": "{not json"},
        }
    )
    assert bad_json.call_id == "call_1"
    assert bad_json.error == {"error": "Tool arguments were not valid JSON"}

    valid = api.inspect_tool_call(
        {
            "id": "call_3",
            "type": "function",
            "function": {"name": "HassTurnOn", "arguments": '{"name": "Kitchen"}'},
        }
    )
    assert valid.error is None
    assert valid.arguments == {"name": "Kitchen"}


def test_task_structure_is_sent_as_venice_response_format_and_a_message() -> None:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    messages: list[dict] = [{"role": "user", "content": "return json"}]
    support.append_structure_instruction(messages, schema)
    assert messages[0]["role"] == "system"
    assert json.loads(messages[0]["content"].split(": ", 1)[1]) == schema
    assert support.venice_response_format(schema) == {
        "type": "json_schema",
        "json_schema": schema,
    }
    assert support.venice_response_format(json.dumps(schema))["json_schema"] == schema
    assert support.venice_response_format(None) is None


def test_fenced_structured_reply_parses() -> None:
    assert support.parse_structured_response('```json\n{"ok": true}\n```') == {"ok": True}


class _BufferedResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.status_code = 200
        self.headers = {"content-type": "audio/mpeg"}
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"text": "heard"}


class _StreamingResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.status_code = 200
        self.closed = False
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _RecordingHttp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.sent: list[dict] = []
        self.response = _StreamingResponse([b"abc", b"def"])

    def build_request(self, method, url, **kwargs):
        self.calls.append((method, {"url": url, **kwargs}))
        return {"method": method, "url": url}

    async def send(self, request, stream: bool = False):
        self.sent.append({"stream": stream})
        return self.response

    async def request(self, method, url, **kwargs):
        self.calls.append((method, {"url": url, **kwargs}))
        return _BufferedResponse(b"audio-bytes")


@pytest.mark.asyncio
async def test_audio_endpoints_sanitize_auth_and_stream() -> None:
    http = _RecordingHttp()
    api_client = client.AsyncVeniceAIClient("sk-abc\nINJECTED", http_client=http)
    audio = await api_client.speech.generate("hello")
    transcript = await api_client.transcriptions.create(b"wav")
    chunks = [chunk async for chunk in api_client.speech.generate_streaming("hello")]

    assert audio == b"audio-bytes"
    assert transcript == {"text": "heard"}
    assert chunks == [b"abc", b"def"]
    assert http.sent == [{"stream": True}]
    assert http.response.closed is True
    authorizations = [call[1]["headers"]["Authorization"] for call in http.calls]
    assert authorizations
    assert set(authorizations) == {"Bearer sk-abcINJECTED"}
