# Contributing to Venice AI Conversation

Thanks for your interest in improving **Venice AI Conversation**! This
document explains how to set up a development environment, run the
test suite, and submit changes.

## Code of Conduct

This project follows the [Home Assistant Community
Forums](https://community.home-assistant.io/) code of conduct: be
respectful, be inclusive, focus on the technical merit of ideas, and
assume good faith.

## Setting up

### Prerequisites

- Python **3.12** (matches the `python_version` declared in
  `hacs.json`).
- A Linux/macOS/Windows shell with `git`, `pip`, and `pytest`.
- An isolated virtual environment is strongly recommended.

### Clone and install

```bash
git clone https://github.com/grasponcrypto/venice_ai.git
cd venice_ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip pytest pytest-asyncio
```

The integration targets Home Assistant; the test suite is designed to
run **without** HA installed (helpers are AST-extracted where
needed). If you want to load the integration inside a development HA
install, copy `custom_components/venice_ai/` into your HA
`config/custom_components/` directory.

## Running the tests

```bash
python -m pytest tests/ -v
```

The current suite has **41 tests** across `test_client.py`,
`test_venice_api.py`, and `test_schema.py`. All tests should pass on
a fresh checkout. The whole run takes well under a second on modern
hardware.

### Adding tests

- Place new tests under `tests/` using the `test_*.py` naming
  convention.
- If you need to test code that imports Home Assistant, either
  AST-extract the pure helper (see `tests/test_schema.py` for the
  pattern) or guard the import with a fixture in `tests/conftest.py`.
- Avoid network calls in tests — use the `FakeChatCompletions`
  fixture for streaming responses and patch `httpx.AsyncClient` for
  HTTP-level tests.

## Project layout

```
venice_ai/
├── custom_components/venice_ai/
│   ├── __init__.py        # Setup, services, migrations
│   ├── client.py          # AsyncVeniceAIClient + retries + metrics
│   ├── venice_api.py      # VeniceConversationService (chat, tools)
│   ├── conversation.py    # HA conversation agent
│   ├── ai_task.py         # AI Task entity
│   ├── tts.py             # TTS entity
│   ├── stt.py             # STT entity
│   ├── sensor.py          # Coordinator + token/last-error sensor
│   ├── config_flow.py     # User/options/reauth flows
│   ├── coordinator.py     # DataUpdateCoordinator wrapper
│   ├── diagnostics.py     # Diagnostics export (redacted)
│   └── const.py           # Constants, defaults, version table
├── tests/
│   ├── conftest.py        # FakeChatCompletions, fixtures
│   ├── test_client.py
│   ├── test_venice_api.py
│   └── test_schema.py
├── hacs.json              # HACS metadata
├── pytest.ini             # Pytest configuration
└── README.md
```

## Style guidelines

- **Type hints everywhere.** Public functions and methods take and
  return annotated types.
- **Docstrings** describe intent, parameters, return value, and any
  raised exceptions. Use the existing module-level docstrings as a
  template.
- **Constants live in `const.py`** when they need to be tunable.
  Don't bury magic numbers in service code.
- **Voluptuous schemas** in `config_flow.py` define the user-facing
  options surface; runtime validation belongs in the entity that
  consumes the values.
- **Logging** uses the per-module `_LOGGER = logging.getLogger(__name__)`.
  Never log the API key, full prompts, or raw tool output.

## Submitting changes

1. **Fork** the repository and create a topic branch
   (`git checkout -b fix/something`).
2. Make focused commits with messages in the form
   `area(scope): short summary`. Examples:
   - `client(retry): cap backoff at RETRY_MAX_DELAY`
   - `docs(readme): add troubleshooting section`
3. Run `python -m pytest tests/ -v` and ensure all tests pass.
4. Push your branch and open a Pull Request against `0.9-revise`.
5. Describe the **what** and the **why** in the PR body. Reference
   any open issue or `CODE_REVIEW_COMPREHENSIVE.md` item by code
   (e.g. `Resolves PERF-3`).
6. Wait for review. A maintainer may request changes before merge.

## Reporting bugs

Use the issue tracker and include:

- Home Assistant version.
- Integration version (visible under **Settings → Devices &
  services → Venice AI Conversation**).
- A redacted debug log snippet
  (`logger: { logs: { "custom_components.venice_ai": debug } }`).
- Reproduction steps and expected vs actual behaviour.

## Release process

1. Update `CHANGELOG.md` under the `[Unreleased]` heading.
2. Bump the version in `custom_components/venice_ai/const.py` and
   the `manifest.json` (if applicable).
3. Tag the release (`git tag -a v0.9.1 -m "..."`).
4. Move the `[Unreleased]` items into a dated heading in
   `CHANGELOG.md`.
5. Push tags (`git push origin --tags`).

## License

By contributing, you agree that your contributions will be licensed
under the same terms as the project (see `LICENSE`).
