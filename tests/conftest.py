"""Shared pytest fixtures and fakes for the Venice AI test suite.

These fakes let the service layer (``venice_api``) be tested in isolation
without a live network or a running Home Assistant instance, satisfying the
TEST-2 goal of integration-style tests with a mocked Venice AI client.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

# Make ``custom_components`` importable when running ``pytest`` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_COMPONENT_DIR = ROOT / "custom_components" / "venice_ai"
_PKG_NAME = "venice_ai_under_test"


def load_component_module(name: str) -> types.ModuleType:
    """Load a single integration module without triggering the package __init__.

    The integration's ``__init__.py`` imports Home Assistant at import time,
    which is not available in a plain unit-test environment. To test the
    pure-logic modules (``client``, ``venice_api``) in isolation we load them
    under a lightweight synthetic package so their relative imports
    (e.g. ``from .client import ...``) resolve correctly.

    Args:
        name: The module name within ``custom_components/venice_ai`` to load,
            e.g. ``"client"`` or ``"venice_api"``.

    Returns:
        The imported module object.
    """
    if _PKG_NAME not in sys.modules:
        pkg = types.ModuleType(_PKG_NAME)
        pkg.__path__ = [str(_COMPONENT_DIR)]  # type: ignore[attr-defined]
        sys.modules[_PKG_NAME] = pkg

    full_name = f"{_PKG_NAME}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(
        full_name, _COMPONENT_DIR / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module



class FakeChunk:
    """Minimal stand-in for ``ChatCompletionChunk`` used by the service layer."""

    def __init__(self, choices: list[dict[str, Any]]) -> None:
        self.choices = choices


class FakeStream:
    """Async-iterable stream of :class:`FakeChunk` objects."""

    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> "FakeStream":
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> FakeChunk:
        try:
            return next(self._iter)
        except StopIteration as err:  # pragma: no cover - trivial
            raise StopAsyncIteration from err


class FakeChatCompletions:
    """Fake of ``client.chat`` supporting streaming and non-streaming calls."""

    def __init__(
        self,
        *,
        chunks: list[FakeChunk] | None = None,
        non_streaming_response: dict[str, Any] | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._non_streaming_response = non_streaming_response or {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        }
        self.last_create_kwargs: dict[str, Any] | None = None
        self.last_non_streaming_kwargs: dict[str, Any] | None = None

    @asynccontextmanager
    async def create(self, **kwargs: Any):
        """Mimic the streaming async-context-manager API."""
        self.last_create_kwargs = kwargs
        yield FakeStream(self._chunks)

    async def create_non_streaming(self, **kwargs: Any) -> dict[str, Any]:
        """Mimic the non-streaming API."""
        self.last_non_streaming_kwargs = kwargs
        return self._non_streaming_response


class FakeClient:
    """Fake ``AsyncVeniceAIClient`` exposing only the ``chat`` namespace."""

    def __init__(self, chat: FakeChatCompletions) -> None:
        self.chat = chat


@pytest.fixture
def make_client():
    """Return a factory building a :class:`FakeClient`."""

    def _factory(
        *,
        chunks: list[FakeChunk] | None = None,
        non_streaming_response: dict[str, Any] | None = None,
    ) -> FakeClient:
        return FakeClient(
            FakeChatCompletions(
                chunks=chunks,
                non_streaming_response=non_streaming_response,
            )
        )

    return _factory


@pytest.fixture
def chunk():
    """Return a helper to build a single-choice :class:`FakeChunk` from a delta."""

    def _build(delta: dict[str, Any]) -> FakeChunk:
        return FakeChunk([{"delta": delta}])

    return _build
