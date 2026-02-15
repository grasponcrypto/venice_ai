from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from collections.abc import AsyncGenerator


def install_homeassistant_stubs() -> None:
    """Install minimal Home Assistant module stubs for Venice tests."""
    if "homeassistant" in sys.modules:
        return

    ha = types.ModuleType("homeassistant")
    ha.__path__ = []

    # homeassistant.const
    ha.const = types.ModuleType("const")
    ha.const.CONF_API_KEY = "api_key"
    ha.const.CONF_LLM_HASS_API = "llm_hass_api"

    # homeassistant.config_entries
    ha.config_entries = types.ModuleType("config_entries")
    ha.config_entries.ConfigFlowResult = dict

    class _BaseConfigFlow:
        def __init_subclass__(cls, *args, domain=None, **kwargs):
            super().__init_subclass__(*args, **kwargs)
            cls.domain = domain

        def async_show_form(self, **kwargs):
            return kwargs

        def async_create_entry(self, **kwargs):
            self.created_entry = kwargs
            return kwargs

    class _OptionsFlow:
        def __init__(self, config_entry):
            self.config_entry = config_entry
            self.hass = None

        def async_show_form(self, **kwargs):
            return kwargs

        def async_create_entry(self, **kwargs):
            self.created_entry = kwargs
            return kwargs

    ha.config_entries.ConfigFlow = _BaseConfigFlow
    ha.config_entries.OptionsFlow = _OptionsFlow
    ha.config_entries.ConfigEntry = object

    # homeassistant.core
    ha.core = types.ModuleType("core")
    ha.core.HomeAssistant = object

    # homeassistant.helpers.*
    ha.helpers = types.ModuleType("helpers")
    ha.helpers.config_validation = types.ModuleType("config_validation")
    ha.helpers.config_validation.string = str

    ha.helpers.llm = types.ModuleType("llm")
    ha.helpers.llm.async_get_apis = lambda hass: []

    ha.helpers.selector = types.ModuleType("selector")

    class _Selector:
        def __init__(self, config=None):
            self.config = config

    class _SelectOptionDict(dict):
        def __init__(self, value, label):
            super().__init__(value=value, label=label)

    class _SelectSelectorConfig:
        def __init__(self, options=None, mode=None, multiple=False):
            self.options = options or []
            self.mode = mode
            self.multiple = multiple

    class _NumberSelectorConfig:
        def __init__(self, min=None, max=None, step=None, mode=None):
            self.min = min
            self.max = max
            self.step = step
            self.mode = mode

    ha.helpers.selector.BooleanSelector = _Selector
    ha.helpers.selector.NumberSelector = _Selector
    ha.helpers.selector.NumberSelectorConfig = _NumberSelectorConfig
    ha.helpers.selector.SelectOptionDict = _SelectOptionDict
    ha.helpers.selector.SelectSelector = _Selector
    ha.helpers.selector.SelectSelectorConfig = _SelectSelectorConfig
    ha.helpers.selector.SelectSelectorMode = types.SimpleNamespace(DROPDOWN="dropdown")
    ha.helpers.selector.TemplateSelector = _Selector

    ha.helpers.entity_platform = types.ModuleType("entity_platform")
    ha.helpers.entity_platform.AddEntitiesCallback = object

    # homeassistant.components.tts
    ha.components = types.ModuleType("components")
    ha.components.__path__ = []
    tts = types.ModuleType("tts")
    tts.__package__ = "homeassistant.components"
    tts.__path__ = []
    tts.ATTR_AUDIO_OUTPUT = "audio_output"
    tts.ATTR_VOICE = "voice"
    tts.TtsAudioType = tuple[str | None, bytes | None]
    tts.TextToSpeechEntity = object

    @dataclass
    class Voice:
        voice_id: str
        name: str

    @dataclass
    class TTSAudioRequest:
        language: str
        options: dict
        message_gen: AsyncGenerator[str, None]

    @dataclass
    class TTSAudioResponse:
        extension: str
        data_gen: AsyncGenerator[bytes, None]

    tts.Voice = Voice
    tts.TTSAudioRequest = TTSAudioRequest
    tts.TTSAudioResponse = TTSAudioResponse
    ha.components.tts = tts

    ha.exceptions = types.ModuleType("exceptions")

    class HomeAssistantError(Exception):
        pass

    ha.exceptions.HomeAssistantError = HomeAssistantError

    # Register modules
    sys.modules["homeassistant"] = ha
    sys.modules["homeassistant.const"] = ha.const
    sys.modules["homeassistant.config_entries"] = ha.config_entries
    sys.modules["homeassistant.core"] = ha.core
    sys.modules["homeassistant.helpers"] = ha.helpers
    sys.modules["homeassistant.helpers.config_validation"] = ha.helpers.config_validation
    sys.modules["homeassistant.helpers.llm"] = ha.helpers.llm
    sys.modules["homeassistant.helpers.selector"] = ha.helpers.selector
    sys.modules["homeassistant.helpers.entity_platform"] = ha.helpers.entity_platform
    sys.modules["homeassistant.components"] = ha.components
    sys.modules["homeassistant.components.tts"] = tts
    sys.modules["homeassistant.exceptions"] = ha.exceptions

    # Third-party dependency stubs used by venice_ai modules.
    if "voluptuous" not in sys.modules:
        vol = types.ModuleType("voluptuous")

        class _Schema:
            def __init__(self, schema):
                self.schema = schema

            def __call__(self, value):
                return value

        class _Marker:
            def __init__(self, key, default=None, description=None):
                self.key = key
                self.default = default
                self.description = description

            def __hash__(self):
                return hash((self.key, self.default))

        vol.Schema = _Schema
        vol.Required = lambda key, default=None, description=None: _Marker(
            key, default=default, description=description
        )
        vol.Optional = lambda key, default=None, description=None: _Marker(
            key, default=default, description=description
        )
        vol.In = lambda values: values
        vol.All = lambda *validators: validators[0] if validators else (lambda x: x)
        vol.Coerce = lambda _type: (lambda x: _type(x))
        vol.Range = lambda min=None, max=None: ("range", min, max)
        vol.Length = lambda min=None, max=None: ("length", min, max)
        sys.modules["voluptuous"] = vol

    if "httpx" not in sys.modules:
        httpx = types.ModuleType("httpx")

        class RequestError(Exception):
            pass

        class HTTPStatusError(Exception):
            def __init__(self, message="", request=None, response=None):
                super().__init__(message)
                self.request = request
                self.response = response

        class AsyncClient:
            async def get(self, *args, **kwargs):
                raise NotImplementedError

            async def post(self, *args, **kwargs):
                raise NotImplementedError

            def stream(self, *args, **kwargs):
                raise NotImplementedError

            async def aclose(self):
                return None

        httpx.RequestError = RequestError
        httpx.HTTPStatusError = HTTPStatusError
        httpx.AsyncClient = AsyncClient
        sys.modules["httpx"] = httpx
