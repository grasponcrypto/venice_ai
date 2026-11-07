"""Regression tests for credential-aware model discovery."""

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from .conftest import load_component_module

client = load_component_module("client")


@pytest.fixture(autouse=True)
def clear_model_cache():
    client._PROCESS_MODEL_CACHE.clear()
    client._MODEL_CACHE_GENERATION.clear()
    yield
    client._PROCESS_MODEL_CACHE.clear()
    client._MODEL_CACHE_GENERATION.clear()


@pytest.mark.asyncio
async def test_cache_is_scoped_and_returns_independent_models():
    requests = []

    def handler(request):
        requests.append((str(request.url), request.headers["authorization"]))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": request.headers["authorization"], "spec": {"voices": ["a"]}}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        first = client.AsyncVeniceAIClient("first", http_client=http)
        same = client.AsyncVeniceAIClient("first", http_client=http)
        other = client.AsyncVeniceAIClient("other", http_client=http)
        endpoint = client.AsyncVeniceAIClient(
            "first", base_url="https://other.example/v1", http_client=http
        )
        models = await first.models.list()
        models[0]["spec"]["voices"].append("changed")
        assert (await same.models.list())[0]["spec"]["voices"] == ["a"]
        assert (await other.models.list())[0]["id"] == "Bearer other"
        await endpoint.models.list()
        await first.models.list("tts")
        assert len(requests) == 4


@pytest.mark.asyncio
async def test_force_refresh_rejects_revoked_key_and_does_not_reuse_stale_cache():
    valid = True
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return (
            httpx.Response(200, json={"data": [{"id": "model"}]})
            if valid
            else httpx.Response(401, text="invalid")
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        first = client.AsyncVeniceAIClient("key", http_client=http)
        assert await first.models.list() == [{"id": "model"}]
        valid = False
        with pytest.raises(client.AuthenticationError):
            await first.models.list(force_refresh=True)
        with pytest.raises(client.AuthenticationError):
            await client.AsyncVeniceAIClient("key", http_client=http).models.list()
        assert calls == 3


@pytest.mark.asyncio
async def test_inflight_old_request_cannot_restore_cache_after_failed_validation():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            return httpx.Response(200, json={"data": [{"id": "old"}]})
        return httpx.Response(401, text="invalid")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        instance = client.AsyncVeniceAIClient("key", http_client=http)
        old_request = asyncio.create_task(instance.models.list())
        await started.wait()
        with pytest.raises(client.AuthenticationError):
            await instance.models.list(force_refresh=True)
        release.set()
        assert await old_request == [{"id": "old"}]
        with pytest.raises(client.AuthenticationError):
            await instance.models.list()
        assert calls == 3


@pytest.mark.asyncio
async def test_cache_expiry_and_successful_refresh(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(client.time, "monotonic", lambda: now[0])
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": [{"id": calls}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        instance = client.AsyncVeniceAIClient("key", http_client=http)
        assert (await instance.models.list())[0]["id"] == 1
        assert (await instance.models.list(force_refresh=True))[0]["id"] == 2
        now[0] += client._PROCESS_MODEL_CACHE_TTL
        assert (await instance.models.list())[0]["id"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["async_step_user", "async_step_reauth_confirm"])
async def test_config_validation_rejects_cached_revoked_key(method):
    source = Path("custom_components/venice_ai/config_flow.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    flow = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VeniceAIConfigFlow"
    )
    function = next(
        node
        for node in flow.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == method
    )
    valid = True

    def handler(request):
        return (
            httpx.Response(200, json={"data": []})
            if valid
            else httpx.Response(401, text="invalid")
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await client.AsyncVeniceAIClient("key", http_client=http).models.list()
        valid = False
        namespace = {
            "AsyncVeniceAIClient": client.AsyncVeniceAIClient,
            "AuthenticationError": client.AuthenticationError,
            "VeniceAIError": client.VeniceAIError,
            "get_async_client": lambda hass: http,
            "CONF_API_KEY": "api_key",
            "_LOGGER": logging.getLogger(__name__),
            "vol": SimpleNamespace(
                Schema=lambda value: value, Required=lambda value: value
            ),
            "cv": SimpleNamespace(string=str),
            "STEP_USER_DATA_SCHEMA": {},
        }
        code = compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "config_flow.py",
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        )
        exec(code, namespace)  # noqa: S102
        flow_stub = SimpleNamespace(
            hass=None,
            _get_reauth_entry=lambda: SimpleNamespace(title="test"),
            async_show_form=lambda **kwargs: kwargs,
        )
        result = await namespace[method](flow_stub, {"api_key": "key"})
        assert result["errors"] == {"base": "invalid_auth"}


@pytest.mark.asyncio
async def test_entry_setup_rejects_cached_revoked_key():
    source = Path("custom_components/venice_ai/__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "async_setup_entry"
    )
    valid = True

    def handler(request):
        return (
            httpx.Response(200, json={"data": []})
            if valid
            else httpx.Response(401, text="invalid")
        )

    class ConfigEntryAuthFailed(Exception):
        pass

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await client.AsyncVeniceAIClient("key", http_client=http).models.list()
        valid = False
        namespace = {
            "AsyncVeniceAIClient": client.AsyncVeniceAIClient,
            "AuthenticationError": client.AuthenticationError,
            "VeniceAIError": client.VeniceAIError,
            "ConfigEntryAuthFailed": ConfigEntryAuthFailed,
            "get_async_client": lambda hass: http,
            "CONF_API_KEY": "api_key",
        }
        code = compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "__init__.py",
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        )
        exec(code, namespace)  # noqa: S102
        with pytest.raises(ConfigEntryAuthFailed):
            await namespace["async_setup_entry"](
                None, SimpleNamespace(data={"api_key": "key"})
            )
