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
