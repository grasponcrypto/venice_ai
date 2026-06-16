# Contributing to Venice AI for Home Assistant

Thank you for your interest in contributing! This document outlines the development workflow, coding standards, and submission guidelines for the `venice_ai` custom integration.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Coding Standards](#coding-standards)
4. [Running Tests](#running-tests)
5. [Submitting Changes](#submitting-changes)
6. [Reporting Issues](#reporting-issues)

---

## Development Setup

### Prerequisites

- Python 3.12+
- Home Assistant development environment (see [HA Developer Docs](https://developers.home-assistant.io/))
- `pip` with a virtual environment recommended

### Install Dependencies

```bash
pip install -r requirements_test.txt
```

If `requirements_test.txt` does not exist yet, install manually:

```bash
pip install pytest pytest-asyncio homeassistant voluptuous voluptuous-openapi aiohttp httpx
```

### Environment Variables

No special environment variables are required for local development. API key validation is handled by the config flow and is not needed for unit tests.

---

## Project Structure

```
custom_components/venice_ai/
├── __init__.py          # Integration setup, services, repair issues
├── client.py            # Async HTTP client wrapping the Venice AI REST API
├── config_flow.py       # Config & options UI flow
├── const.py             # All shared constants and feature flags
├── conversation.py      # ConversationEntity — LLM chat with tool support
├── coordinator.py       # DataUpdateCoordinator — model & voice metadata
├── sensor.py            # Diagnostic sensors (model count, etc.)
├── tts.py               # Text-to-Speech platform
├── stt.py               # Speech-to-Text platform (ASR)
├── ai_task.py           # AI Task platform (HA 2024.8+)
└── venice_api.py        # Service-layer helpers (ChatParameters, VeniceConversationService)

tests/
├── conftest.py          # Shared fixtures
├── test_client.py       # Unit tests for AsyncVeniceAIClient
└── test_venice_api.py   # Unit tests for VeniceConversationService
```

---

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) and Home Assistant's [development guidelines](https://developers.home-assistant.io/docs/development_guidelines).
- All new code must be typed — use `from __future__ import annotations` and full type hints.
- Prefer `_LOGGER.debug` for verbose output; `_LOGGER.warning` / `_LOGGER.error` for actionable issues.
- Avoid blocking I/O on the event loop. Use `await` for all network calls.
- Constants belong in `const.py`. Do not hard-code magic numbers or strings elsewhere.
- New config/options keys must be added to both `const.py` and `config_flow.py`.

---

## Running Tests

```bash
cd c:/path/to/venice_ai
pytest tests/ -v
```

To run a specific test file:

```bash
pytest tests/test_client.py -v
```

To check coverage:

```bash
pytest tests/ --cov=custom_components/venice_ai --cov-report=term-missing
```

---

## Submitting Changes

1. Fork the repository and create a feature branch: `git checkout -b feature/my-improvement`
2. Make your changes and add or update tests as appropriate.
3. Run the full test suite and ensure all tests pass.
4. Commit with a clear message: `git commit -m "feat: add X to improve Y"`
5. Open a Pull Request against `main`. Include a summary of changes and reference any related issues.

### Commit Message Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/) style:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code change that is not a fix or feature
- `test:` — adding or updating tests
- `chore:` — maintenance tasks

---

## Reporting Issues

Please open a GitHub issue with:
- A clear description of the bug or feature request.
- Your Home Assistant version and integration version.
- Relevant log output (set logger level to `debug` for `custom_components.venice_ai`).
