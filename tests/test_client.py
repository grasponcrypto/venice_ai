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
