"""Unit tests for ``client.py`` pure logic (TEST-1).

Covers the telemetry counters backing the diagnostic sensors (LOW-4) and the
centralised HTTP error categorization that gives every API method consistent,
typed exceptions.
"""

from __future__ import annotations

import pytest

from .conftest import load_component_module

client = load_component_module("client")


class TestVeniceAIMetrics:
    """Tests for the :class:`VeniceAIMetrics` telemetry dataclass."""

    def test_initial_state_is_zeroed(self) -> None:
        metrics = client.VeniceAIMetrics()
        assert metrics.request_count == 0
        assert metrics.error_count == 0
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0
        assert metrics.last_error is None

    def test_record_request_increments(self) -> None:
        metrics = client.VeniceAIMetrics()
        metrics.record_request()
        metrics.record_request()
        assert metrics.request_count == 2

    def test_record_error_tracks_count_and_message(self) -> None:
        metrics = client.VeniceAIMetrics()
        metrics.record_error(ValueError("boom"))
        assert metrics.error_count == 1
        assert metrics.last_error == "ValueError: boom"

    def test_record_usage_accumulates(self) -> None:
        metrics = client.VeniceAIMetrics()
        metrics.record_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        metrics.record_usage(
            {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        )
        assert metrics.prompt_tokens == 13
        assert metrics.completion_tokens == 7
        assert metrics.total_tokens == 20

    def test_record_usage_ignores_non_dict(self) -> None:
        metrics = client.VeniceAIMetrics()
        metrics.record_usage(None)
        metrics.record_usage("not-a-dict")  # type: ignore[arg-type]
        assert metrics.total_tokens == 0

    def test_record_usage_tolerates_missing_and_none_fields(self) -> None:
        metrics = client.VeniceAIMetrics()
        metrics.record_usage({"prompt_tokens": None})
        assert metrics.prompt_tokens == 0
        assert metrics.total_tokens == 0

    def test_listeners_fire_on_change_and_can_unsubscribe(self) -> None:
        metrics = client.VeniceAIMetrics()
        seen: list[int] = []
        unsubscribe = metrics.add_listener(lambda: seen.append(metrics.request_count))
        metrics.record_request()
        unsubscribe()
        metrics.record_request()
        metrics.record_usage(None)
        assert seen == [1]
        assert metrics.request_count == 2

    def test_listener_failure_does_not_break_recording(self) -> None:
        metrics = client.VeniceAIMetrics()

        def _boom() -> None:
            raise RuntimeError("sensor gone")

        metrics.add_listener(_boom)
        metrics.record_error(ValueError("boom"))
        assert metrics.error_count == 1


class TestErrorCategorization:
    """Tests for ``_categorize_http_error`` status-code mapping."""

    def test_401_is_authentication_error(self) -> None:
        err = client._categorize_http_error(401, "nope", "fetching models")
        assert isinstance(err, client.AuthenticationError)

    def test_429_is_rate_limit_error(self) -> None:
        err = client._categorize_http_error(429, "slow down")
        assert isinstance(err, client.RateLimitError)

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_5xx_is_service_unavailable(self, status: int) -> None:
        err = client._categorize_http_error(status, "down")
        assert isinstance(err, client.ServiceUnavailableError)

    def test_generic_4xx_is_base_error(self) -> None:
        err = client._categorize_http_error(418, "teapot")
        assert isinstance(err, client.VeniceAIError)

    def test_context_included_in_message(self) -> None:
        err = client._categorize_http_error(429, "limit", "during chat")
        assert "during chat" in str(err)


class TestSanitizeHeaderValue:
    """Tests for ``_sanitize_header_value`` (SEC-1).

    Regression guard: commit 64b115c implemented this helper with a
    ``.strip()`` + ``ord(ch) >= 0x20`` filter, which silently mutated
    API keys that had surrounding whitespace into byte-different strings.
    Those mutated keys authenticated against Venice as "invalid" and the
    integration began logging HTTP 401 on every request even though the
    user's stored key was valid. These tests pin the contract: only
    ``\\r`` and ``\\n`` are removed, everything else passes through.
    """

    def test_none_returns_empty_string(self) -> None:
        assert client._sanitize_header_value(None) == ""

    def test_empty_string_returns_empty_string(self) -> None:
        assert client._sanitize_header_value("") == ""

    def test_plain_ascii_key_is_preserved(self) -> None:
        key = "sk-AbCdEfGh1234567890ZyXwVuTsRqPo"
        assert client._sanitize_header_value(key) == key

    def test_leading_and_trailing_whitespace_preserved(self) -> None:
        # Regression: the old .strip() implementation removed these and
        # produced a byte-different key, which Venice rejected with 401.
        # We must keep the key byte-for-byte intact (httpx handles any
        # well-defined trailing-whitespace trimming on the wire itself).
        key = "  sk-AbCdEfGh1234567890  "
        assert client._sanitize_header_value(key) == "  sk-AbCdEfGh1234567890  "

    def test_internal_tab_preserved(self) -> None:
        # Tabs inside a credential are unusual but the SEC-1 contract is
        # "remove only CR/LF" — we must not silently edit other bytes.
        key = "sk-\tabc"
        assert client._sanitize_header_value(key) == "sk-\tabc"

    def test_trailing_newline_stripped(self) -> None:
        # The actual header-injection vector: a stray \n at the end of a
        # pasted API key. This MUST be removed.
        key = "sk-abc123\n"
        assert client._sanitize_header_value(key) == "sk-abc123"

    def test_carriage_return_stripped(self) -> None:
        key = "sk-abc123\r"
        assert client._sanitize_header_value(key) == "sk-abc123"

    def test_crlf_inside_key_stripped(self) -> None:
        # The classic header-injection payload: CRLF followed by a fake
        # header. Only the CR/LF bytes are removed; the rest is passed
        # through (and httpx will reject it on the wire if it is still
        # malformed, which is the correct defense-in-depth posture).
        key = "sk-abc\r\nX-Evil-Header: injected"
        out = client._sanitize_header_value(key)
        assert "\r" not in out
        assert "\n" not in out
        assert out == "sk-abcX-Evil-Header: injected"

    def test_nbsp_preserved(self) -> None:
        # Non-ASCII whitespace (\xa0) is NOT Python-defined whitespace for
        # the purposes of this function. The old implementation's .strip()
        # would remove it on some Python builds and turn a valid key into
        # a 401-rejected one. Pin: leave it alone.
        key = "\xa0sk-abc123\xa0"
        assert client._sanitize_header_value(key) == "\xa0sk-abc123\xa0"

    def test_non_ascii_key_bytes_preserved(self) -> None:
        # If Venice ever issues keys with non-ASCII characters, we must
        # not filter them out. The old ord >= 0x20 check happened to allow
        # these, but combined with .strip() the function still broke
        # surrounding whitespace. Pin the full contract here.
        key = "sk-café-ñ-ü"
        assert client._sanitize_header_value(key) == "sk-café-ñ-ü"

    def test_stored_api_key_is_unmodified(self) -> None:
        """The client stores the raw key; only the header value is scrubbed.

        Regression: commit 64b115c stored ``safe_api_key`` on
        ``self._api_key``, so any code path that re-used the in-memory
        key (diagnostics, re-auth round-trip, the Speech/Transcriptions
        per-request headers) would re-mutate an already-mutated value.
        """
        import httpx
        raw_key = "  sk-AbCdEfGh1234567890  \n"
        c = client.AsyncVeniceAIClient(api_key=raw_key, http_client=httpx.AsyncClient())
        try:
            # In-memory copy is byte-for-byte the user-provided value.
            assert c._api_key == raw_key
            # Header value has only CR/LF removed — surrounding whitespace
            # is preserved so Venice sees the same key the user entered.
            assert c._headers["Authorization"] == f"Bearer {raw_key.replace(chr(10), '')}"
            assert "\n" not in c._headers["Authorization"]
        finally:
            import asyncio
            asyncio.get_event_loop_policy()
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(c.close())
            finally:
                loop.close()


PIXEL = {"widthHeightDivisor": 8, "promptCharacterLimit": 20}
ASPECT = {
    "aspectRatios": ["1:1", "16:9"],
    "widthHeightDivisor": 8,
    "promptCharacterLimit": 100,
}
RESOLUTION = {
    "resolutions": ["1K", "2K"],
    "aspectRatios": ["1:1"],
    "widthHeightDivisor": 8,
}


class TestImageGenerateOptions:
    """Sizing rules for native POST /image/generate."""

    def test_pixel_model_keeps_divisible_edges(self) -> None:
        options = client.image_generate_options(
            PIXEL, "a cat", width=1024, height=768
        )
        assert options["width"] == 1024
        assert options["height"] == 768
        assert options["format"] == "webp"
        assert "aspect_ratio" not in options

    def test_pixel_model_rejects_bad_divisor(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="divisible by 8"):
            client.image_generate_options(PIXEL, "a cat", width=1004, height=768)

    def test_pixel_model_rejects_edge_above_max(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="between 1 and 1280"):
            client.image_generate_options(PIXEL, "a cat", width=1288, height=768)

    def test_pixel_model_rejects_aspect_ratio(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="aspect_ratio"):
            client.image_generate_options(PIXEL, "a cat", aspect_ratio="1:1")

    def test_pixel_model_requires_both_edges(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="together"):
            client.image_generate_options(PIXEL, "a cat", width=1024)

    def test_aspect_model_keeps_listed_ratio(self) -> None:
        options = client.image_generate_options(
            ASPECT, "a cat", aspect_ratio="16:9"
        )
        assert options["aspect_ratio"] == "16:9"
        assert "width" not in options

    def test_aspect_model_rejects_unknown_ratio(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="one of"):
            client.image_generate_options(ASPECT, "a cat", aspect_ratio="4:3")

    def test_aspect_model_rejects_pixels(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="width or height"):
            client.image_generate_options(
                ASPECT, "a cat", width=1024, height=1024
            )

    def test_resolution_model_keeps_listed_tier(self) -> None:
        options = client.image_generate_options(
            RESOLUTION, "a cat", aspect_ratio="1:1", resolution="2K"
        )
        assert options["resolution"] == "2K"
        assert options["aspect_ratio"] == "1:1"

    def test_resolution_model_rejects_unknown_tier(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="resolution"):
            client.image_generate_options(RESOLUTION, "a cat", resolution="8K")

    def test_empty_constraints_reject_sizing(self) -> None:
        options = client.image_generate_options({}, "a cat")
        assert options == {"prompt": "a cat", "format": "webp"}
        with pytest.raises(client.ImageGenerateError):
            client.image_generate_options({}, "a cat", width=1024, height=1024)

    def test_blank_prompt_rejected(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="Prompt is required"):
            client.image_generate_options(PIXEL, "   ")

    def test_prompt_over_limit_rejected(self) -> None:
        with pytest.raises(client.ImageGenerateError, match="allows 20"):
            client.image_generate_options(PIXEL, "x" * 21)

    def test_find_image_model_skips_video(self) -> None:
        models = [
            {"id": "video-one", "type": "video"},
            {"id": "flux", "type": "image", "model_spec": {"constraints": PIXEL}},
        ]
        found = client.find_image_model(models, "flux")
        assert found is not None
        assert client.image_model_constraints(found)["widthHeightDivisor"] == 8
        assert client.find_image_model(models, "video-one") is None
        assert client.find_image_model(models, "default") is None


class _FakeImageResponse:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self._payload = payload
        self.text = "{}"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeImageHttp:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs.get("json")))
        return _FakeImageResponse({"id": "img-1", "images": ["abc"]})


class TestNativeImageGenerate:
    """The client posts to the native endpoint and omits unused sizing."""

    async def test_posts_native_endpoint_without_openai_fields(self) -> None:
        http = _FakeImageHttp()
        api = client.AsyncVeniceAIClient("key", http_client=http)
        result = await api.images.generate(
            "flux",
            "a cat",
            constraints=PIXEL,
            width=1024,
            height=768,
        )
        method, url, payload = http.calls[0]
        assert method == "POST"
        assert url.endswith("/image/generate")
        assert payload == {
            "model": "flux",
            "prompt": "a cat",
            "format": "webp",
            "width": 1024,
            "height": 768,
        }
        assert "size" not in payload
        assert "quality" not in payload
        assert "style" not in payload
        assert result["images"] == ["abc"]
        assert api.metrics.request_count == 1
        assert api.metrics.error_count == 0

    async def test_default_model_is_rejected_before_the_request(self) -> None:
        http = _FakeImageHttp()
        api = client.AsyncVeniceAIClient("key", http_client=http)
        with pytest.raises(client.ImageGenerateError, match="default"):
            await api.images.generate("default", "a cat")
        assert http.calls == []


class _FakeStreamResponse:
    """Minimal httpx response for ``ChatCompletions.create``."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines
        self.text = ""
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                "boom",
                request=httpx.Request("POST", "https://api.venice.ai/api/v1/chat/completions"),
                response=httpx.Response(self.status_code, request=httpx.Request("POST", "https://api.venice.ai")),
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


class _FakeStreamHttp:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self.response = response

    def build_request(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    async def send(self, request, stream: bool = False):
        return self.response


class TestStreamingChatMetrics:
    """Streaming chat is the default path and must move the diagnostic sensors."""

    async def test_stream_records_request_and_usage(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}',
            "data: [DONE]",
        ]
        http = _FakeStreamHttp(_FakeStreamResponse(lines))
        api = client.AsyncVeniceAIClient("key", http_client=http)

        async with api.chat.create(model="m", messages=[{"role": "user", "content": "hi"}]) as stream:
            chunks = [chunk async for chunk in stream]

        assert len(chunks) == 2
        assert api.metrics.request_count == 1
        assert api.metrics.prompt_tokens == 4
        assert api.metrics.completion_tokens == 2
        assert api.metrics.total_tokens == 6
        assert api.metrics.error_count == 0
        assert http.response.closed is True


class _RetryResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.read = False
        self.closed = False

    async def aread(self) -> bytes:
        self.read = True
        return b""

    async def aclose(self) -> None:
        self.closed = True


class _BrokenReadResponse(_RetryResponse):
    async def aread(self) -> bytes:
        raise RuntimeError("truncated")


class _RetryHttp:
    def __init__(self, responses: list[_RetryResponse]) -> None:
        self.responses = responses
        self.calls = 0

    async def request(self, method: str, url: str, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class TestRetryReleasesResponse:
    async def test_failed_attempt_is_closed_before_retry(self, monkeypatch) -> None:
        async def _no_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(client.asyncio, "sleep", _no_sleep)
        failed = _RetryResponse(503)
        ok = _RetryResponse(200)
        http = _RetryHttp([failed, ok])
        api = client.AsyncVeniceAIClient("key", http_client=http)

        response = await api._async_request_with_retry("GET", "/models")

        assert response is ok
        assert failed.read is True
        assert failed.closed is True
        assert ok.closed is False

    async def test_close_still_runs_when_body_read_fails(self, monkeypatch) -> None:
        async def _no_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(client.asyncio, "sleep", _no_sleep)
        failed = _BrokenReadResponse(429)
        ok = _RetryResponse(200)
        http = _RetryHttp([failed, ok])
        api = client.AsyncVeniceAIClient("key", http_client=http)

        response = await api._async_request_with_retry("GET", "/models")

        assert response is ok
        assert failed.closed is True
        assert ok.closed is False

