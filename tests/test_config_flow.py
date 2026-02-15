import importlib
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from hass_stubs import install_homeassistant_stubs

install_homeassistant_stubs()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)


def _install_venice_namespace_package() -> None:
    """Avoid importing integration __init__.py during unit tests."""
    if "custom_components" not in sys.modules:
        pkg = types.ModuleType("custom_components")
        pkg.__path__ = [os.path.join(BASE_DIR, "custom_components")]
        sys.modules["custom_components"] = pkg

    if "custom_components.venice_ai" not in sys.modules:
        pkg = types.ModuleType("custom_components.venice_ai")
        pkg.__path__ = [os.path.join(BASE_DIR, "custom_components", "venice_ai")]
        sys.modules["custom_components.venice_ai"] = pkg


_install_venice_namespace_package()
cfg_flow = importlib.import_module("custom_components.venice_ai.config_flow")


class DummyModels:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else [{"id": "m1"}]
        self.error = error

    async def list(self):
        if self.error:
            raise self.error
        return self.result


class DummyClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.models = DummyModels()


class DummyAPI:
    def __init__(self, api_id, name):
        self.id = api_id
        self.name = name


class DummyConfigEntry:
    def __init__(self, options=None, runtime_data=None):
        self.options = options or {}
        self.runtime_data = runtime_data


@pytest.mark.asyncio
async def test_config_flow_user_step_creates_entry(monkeypatch):
    monkeypatch.setattr(cfg_flow, "AsyncVeniceAIClient", DummyClient)
    flow = cfg_flow.VeniceAIConfigFlow()

    result = await flow.async_step_user({"api_key": "k"})

    assert result["title"] == "Venice AI"
    assert result["data"]["api_key"] == "k"
    assert result["options"]["chat_model"] == cfg_flow.RECOMMENDED_CHAT_MODEL
    assert result["options"]["llm_hass_api"] == []
    assert "prompt" in result["options"]


@pytest.mark.asyncio
async def test_config_flow_user_step_invalid_auth(monkeypatch):
    class FailingClient:
        def __init__(self, api_key):
            self.models = DummyModels(error=cfg_flow.AuthenticationError("nope"))

    monkeypatch.setattr(cfg_flow, "AsyncVeniceAIClient", FailingClient)
    flow = cfg_flow.VeniceAIConfigFlow()

    result = await flow.async_step_user({"api_key": "bad"})

    assert result["step_id"] == "user"
    assert result["errors"]["base"] == "invalid_auth"


@pytest.mark.asyncio
async def test_options_flow_sanitizes_llm_api_ids(monkeypatch):
    apis = [DummyAPI("valid_api", "Valid API"), DummyAPI("other_api", "Other API")]
    monkeypatch.setattr(cfg_flow.llm, "async_get_apis", lambda hass: apis)

    entry = DummyConfigEntry(options={"llm_hass_api": ["stale_api"]}, runtime_data=None)
    flow = cfg_flow.VeniceAIOptionsFlow(entry)
    flow.hass = object()
    # VeniceAIOptionsFlow.__init__ does not call super(), so set this explicitly
    # for the lightweight stub environment.
    flow.config_entry = entry

    result = await flow.async_step_init({"llm_hass_api": ["valid_api", "unknown_api"]})

    assert result["data"]["llm_hass_api"] == ["valid_api"]
