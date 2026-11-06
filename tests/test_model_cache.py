"""Model-list cache is per API key and can be bypassed for auth checks."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from .conftest import load_component_module

client = load_component_module("client")


class _Response:
    def __init__(self, status: int = 200, models: list[dict] | None = None, text: str = "") -> None:
        self.status_code = status
        self.text = text
        self._models = models or []

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        request = httpx.Request("GET", "https://api.venice.ai/api/v1/models")
        response = httpx.Response(self.status_code, request=request, text=self.text)
        raise httpx.HTTPStatusError("failed", request=request, response=response)

    def json(self) -> dict:
        return {"data": self._models}


class _FakeClient:
    def __init__(self, api_key: str, models: list[dict] | None = None, status: int = 200) -> None:
        self._api_key = api_key
        self._base_url = "https://api.venice.ai/api/v1"
        self._headers = {}
        self.calls = 0
        self._models = models or [{"id": api_key}]
        self._status = status

    async def _async_request_with_retry(self, *args, **kwargs):
        self.calls += 1
        return _Response(self._status, self._models, text="denied")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _clear_model_cache():
    client._PROCESS_MODEL_CACHE.clear()
    yield
    client._PROCESS_MODEL_CACHE.clear()


def test_same_key_reuses_process_cache() -> None:
    first = _FakeClient("key-a", [{"id": "model-a"}])
    second = _FakeClient("key-a", [{"id": "should-not-fetch"}])
    listed = _run(client.Models(first).list())
    again = _run(client.Models(second).list())
    assert listed == [{"id": "model-a"}]
    assert again == [{"id": "model-a"}]
    assert first.calls == 1
    assert second.calls == 0


def test_different_keys_do_not_share_a_model_list() -> None:
    first = _FakeClient("key-a", [{"id": "model-a"}])
    second = _FakeClient("key-b", [{"id": "model-b"}])
    assert _run(client.Models(first).list()) == [{"id": "model-a"}]
    assert _run(client.Models(second).list()) == [{"id": "model-b"}]
    assert second.calls == 1


def test_force_refresh_hits_the_api_even_when_the_cache_is_warm() -> None:
    api = _FakeClient("key-a", [{"id": "model-a"}])
    models = client.Models(api)
    _run(models.list())
    _run(models.list(force_refresh=True))
    assert api.calls == 2


def test_auth_failure_drops_the_cached_list_for_that_key() -> None:
    good = _FakeClient("key-a", [{"id": "model-a"}])
    _run(client.Models(good).list())
    assert client._PROCESS_MODEL_CACHE

    denied = _FakeClient("key-a", status=401)
    with pytest.raises(client.AuthenticationError):
        _run(client.Models(denied).list(force_refresh=True))
    assert client._PROCESS_MODEL_CACHE == {}
