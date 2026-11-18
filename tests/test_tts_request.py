"""Speech requests must not ask Venice for a format or voice it will reject."""

from __future__ import annotations

from .conftest import load_component_module

catalog = load_component_module("model_catalog")
client = load_component_module("client")

KOKORO = {
    "id": "tts-kokoro",
    "model_type": "tts",
    "model_spec": {
        "voices": ["af_sky", "bm_daniel"],
        "supported_formats": ["mp3", "opus", "wav"],
        "default_format": "mp3",
    },
}
ORPHEUS = {
    "id": "tts-orpheus",
    "model_type": "tts",
    "model_spec": {
        "voices": ["tara", "leo"],
        "supported_formats": ["wav"],
        "default_format": "wav",
        "default_voice": "tara",
    },
}
XAI = {
    "id": "tts-xai-v1",
    "model_type": "tts",
    "model_spec": {
        "voices": ["eve", "ara"],
        "supported_formats": ["mp3", "wav", "pcm"],
        "default_format": "mp3",
    },
}
ELEVEN = {
    "id": "tts-elevenlabs-turbo-v2-5",
    "model_type": "tts",
    "model_spec": {
        "voices": ["Rachel"],
        "supported_formats": ["mp3"],
        "default_format": "mp3",
        "supports_custom_voice_id": True,
    },
}
MODELS = [KOKORO, ORPHEUS, XAI, ELEVEN]


def _resolve(**kwargs):
    params = {
        "text": "Done. The office light is off.",
        "model": "tts-kokoro",
        "voice": "bm_daniel",
        "configured_voice": "bm_daniel",
        "response_format": "mp3",
        "configured_format": "mp3",
        "speed": 1.0,
        "models": MODELS,
        "catalog_ok": True,
        "fallback_model": "tts-kokoro",
        "fallback_voice": "bm_daniel",
    }
    params.update(kwargs)
    return catalog.resolve_tts_speech(**params)


def test_wav_only_model_does_not_keep_mp3() -> None:
    resolved = _resolve(model="tts-orpheus", voice="tara", configured_voice="tara")
    assert resolved.response_format == "wav"
    assert resolved.model == "tts-orpheus"
    assert any("mp3" in note for note in resolved.notes)


def test_pipeline_voice_falls_back_to_configured_voice() -> None:
    resolved = _resolve(
        model="tts-xai-v1",
        voice="bm_daniel",
        configured_voice="eve",
    )
    assert resolved.voice == "eve"
    assert resolved.model == "tts-xai-v1"
    assert any("configured voice" in note for note in resolved.notes)


def test_voice_match_is_case_insensitive() -> None:
    resolved = _resolve(model="tts-xai-v1", voice="Eve", configured_voice="ara")
    assert resolved.voice == "eve"


def test_saved_ogg_maps_to_opus_when_the_model_supports_it() -> None:
    resolved = _resolve(response_format="ogg", configured_format="ogg")
    assert resolved.response_format == "opus"


def test_ogg_on_mp3_only_model_uses_model_default() -> None:
    resolved = _resolve(
        model="tts-elevenlabs-turbo-v2-5",
        voice="Rachel",
        configured_voice="Rachel",
        response_format="ogg",
        configured_format="ogg",
    )
    assert resolved.response_format == "mp3"


def test_cloned_voice_handle_is_kept() -> None:
    resolved = _resolve(voice="vv_office", configured_voice="bm_daniel")
    assert resolved.voice == "vv_office"


def test_custom_voice_id_is_kept_when_the_model_allows_it() -> None:
    resolved = _resolve(
        model="tts-elevenlabs-turbo-v2-5",
        voice="provider-voice-id",
        configured_voice="Rachel",
    )
    assert resolved.voice == "provider-voice-id"


def test_missing_model_uses_fallback_and_a_valid_voice() -> None:
    resolved = _resolve(model="tts-retired", voice="nope", configured_voice="nope")
    assert resolved.model == "tts-kokoro"
    assert resolved.voice == "af_sky"


def test_catalog_miss_does_not_rewrite_a_known_pair_except_invalid_format() -> None:
    resolved = _resolve(
        catalog_ok=False,
        model="tts-orpheus",
        voice="tara",
        response_format="ogg",
        configured_format="mp3",
    )
    assert resolved.model == "tts-orpheus"
    assert resolved.voice == "tara"
    assert resolved.response_format == "opus"


def test_speed_string_is_clamped_and_text_is_truncated() -> None:
    resolved = _resolve(speed="9", text="a" * 5000)
    assert resolved.speed == 4.0
    assert len(resolved.text) == catalog.SPEECH_INPUT_LIMIT


def test_audio_labeled_json_error_is_not_hidden() -> None:
    body = b'{"error":"voice bm_daniel is not valid for tts-xai-v1"}'
    detail = client._speech_error_detail(body, "audio/mpeg", 400)
    assert "bm_daniel" in detail
    assert client._streaming_speech_should_retry(detail) is False


def test_unreadable_audio_400_can_be_retried_without_streaming() -> None:
    detail = client._speech_error_detail(b"\xff\xfb\x90\x00" * 40, "audio/mpeg", 400)
    assert detail.startswith("HTTP 400")
    assert client._streaming_speech_should_retry(detail) is True
    assert client._streaming_speech_should_retry(
        "streaming is not supported for this model"
    )


async def test_streaming_400_retries_without_the_streaming_flag() -> None:
    import httpx

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode()
        bodies.append({"url": str(request.url), "body": payload})
        if '"streaming": true' in payload or '"streaming":true' in payload:
            return httpx.Response(
                400,
                content=b'{"error":"streaming is not supported"}',
                headers={"content-type": "audio/mpeg"},
            )
        return httpx.Response(200, content=b"ID3audio")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        venice = client.AsyncVeniceAIClient("key", http_client=http)
        chunks = [
            chunk
            async for chunk in venice.speech.generate_streaming(
                "Done. The office light is off.",
                voice="tara",
                model="tts-orpheus",
                audio_output="wav",
            )
        ]
    assert chunks == [b"ID3audio"]
    assert len(bodies) == 2
    assert "streaming" not in bodies[1]["body"]


async def test_speech_validation_400_is_exposed_and_not_retried() -> None:
    import json

    import httpx
    import pytest

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append({"url": str(request.url), "body": request.read().decode()})
        return httpx.Response(
            400,
            content=b'{"error":{"message":"Invalid voice for model"}}',
            headers={"content-type": "audio/mpeg"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        venice = client.AsyncVeniceAIClient("key", http_client=http)
        with pytest.raises(client.VeniceAIError) as caught:
            _ = [
                chunk
                async for chunk in venice.speech.generate_streaming(
                    "Done.",
                    voice="bm_daniel",
                    model="tts-qwen3-1-7b",
                    audio_output="mp3",
                )
            ]

    assert len(bodies) == 1
    assert json.loads(bodies[0]["body"])["streaming"] is True
    assert "Invalid voice for model" in str(caught.value)


async def test_streaming_uses_true_streaming_transport() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"ID3audio")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        venice = client.AsyncVeniceAIClient("key", http_client=http)
        # Prevent a buffered transport response from satisfying a regression to
        # the retry request helper without exercising the true streaming path.
        async def fail_buffered_request(*args, **kwargs):
            raise AssertionError("streaming speech used the buffered helper")

        venice._async_request_with_retry = fail_buffered_request
        chunks = [chunk async for chunk in venice.speech.generate_streaming("Done.")]
    assert chunks == [b"ID3audio"]
