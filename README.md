# Venice AI
Home Assistant Venice AI Conversation Integration

## Overview
The Venice AI integration allows you to enhance your Home Assistant setup with advanced conversational capabilities powered by Venice AI. This integration enables seamless interaction with your smart home devices through natural language processing.

## Features
- Natural language understanding for smart home commands
- Customizable responses based on user preferences
- Integration with various Home Assistant components
- Dynamic model selection from available Venice AI models

## Installation

### Option 1: HACS (Recommended)
1. Make sure you have [HACS](https://hacs.xyz/) installed in your Home Assistant instance.
2. Click on HACS in the sidebar.
3. Go to "Integrations".
4. Click the three dots in the top right corner and select "Custom repositories".
5. Add `https://github.com/grasponcrypto/venice_ai` as a repository with category "Integration".
6. Click "Add".
7. Search for "Venice AI" in the integrations tab.
8. Click "Download" and follow the installation instructions.
9. Restart Home Assistant.

### Option 2: Manual Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/grasponcrypto/venice_ai.git
   ```

2. **Copy the integration files:**
   Place the `venice_ai` folder in your Home Assistant `custom_components` directory.

3. **Restart Home Assistant:**
   After copying the files, restart your Home Assistant instance to load the new integration.

## Configuration
To configure the Venice AI integration:

1. Go to Settings → Devices & Services
2. Click "Add Integration" and search for "Venice AI"
3. Enter your Venice AI API key
4. Configure additional options as needed:
   - Select your preferred model
   - Adjust temperature, max tokens, and other parameters
   - Customize the system prompt if desired

## Models

The options flow lists every text model returned by Venice AI. It does not filter that list to models that support function calling. Device control still needs a model that can call tools, so choose one that does if you enable Control Home Assistant.

The default chat model is `deepseek-v4-1-flash-1017-uncensored` (DeepSeek V4.1 Flash 1017 Uncensored). Venice marks that model as both `default` and `function_calling_default`. It supports tool calling, structured responses, and a 1M-token context. The previous default, `e2ee-gemma-4-31b`, is no longer in the Venice model list. Existing installs still set to that ID get a repair pointing at the new default.

Thinking is disabled by default so reasoning models stay fast enough for Assist. Turn **Disable thinking** off in the integration options if you want the model to reason before answering.

## Operations

Once the integration is configured, the following surfaces are available:

* **Conversation agent** — A "Venice AI" agent appears in Settings → Voice Assistants → Expose, and can be selected in any Assist pipeline.
* **AI Task entity** — `ai_task.venice_ai_<entry_id>` exposes the model as a structured-data generator for dashboards, scripts, and automations.
* **Text-to-speech** — A `tts.venice_ai_<entry_id>` entity streams synthesized speech from Venice's audio models for use with media players and announce automations.
* **Sensor entity** — `sensor.venice_ai_<entry_id>` reports the latest request count, token usage, and the most recent error message. Useful for dashboarding and HA statistics.
* **Generate image** — `venice_ai.generate_image` creates an image and returns the Venice result. Required fields are the config entry and `prompt`. Optional fields are `size` (`1024x1024`, `1024x1792`, or `1792x1024`; default `1024x1024`), `quality` (`standard` or `hd`; default `standard`), and `style` (`vivid` or `natural`; default `vivid`).
* **AI Task action** — `venice_ai.ai_task` generates structured data and returns `conversation_id` and `data`. It is registered when the AI Task platform is available. Required fields are the config entry and `task`. `structure` is optional and describes the expected response schema.

There is no `venice_ai.refresh_data` action.

### Reconfiguration

Use the integration's "Configure" button to change the API key, model, temperature, max tokens, prompt, or to toggle the "Disable thinking" option for reasoning models. Changes are applied immediately without a restart; in-flight requests will complete using the previous settings.

### Removing the integration

From Settings → Devices & Services → Venice AI, click the three-dot menu and select "Delete". This also removes all associated entities and the AI Task / TTS entities that were created for that entry.

## Troubleshooting

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| `401 Unauthorized` in the log | Invalid or expired Venice AI API key | Update the key via the integration's Configure flow. |
| `429 Rate limit exceeded` | Venice is throttling the account | Lower request concurrency, reduce prompt size, or upgrade the Venice plan. |
| `5xx Service Unavailable` | Venice upstream incident | The integration retries with backoff; verify status on the Venice dashboard before opening an issue. |
| `Connection refused` / `Timeout` | Network egress is blocked from Home Assistant to `api.venice.ai` | Allow outbound HTTPS to `api.venice.ai` on port 443. |
| `No models returned` during setup | Venice returned an empty model list, or the API key cannot list models | Confirm the key is valid and can list models, then retry setup. |
| Agent never responds | Selected model does not support tool calling | Pick a model that supports function calling. The options list includes every text model and is not filtered to tool-capable models. |
| Reasoning model is very slow / verbose | Thinking tokens are being emitted alongside the answer | Enable **Disable thinking** in the integration options to skip reasoning output for faster responses. |
| TTS produces no audio | Selected voice model is unavailable for the chosen language, or audio playback is muted on the target media player | Pick another voice model in the TTS entity options and confirm the target media player is not muted. |

### Diagnostics

1. Enable debug logging for this integration by adding the following to `configuration.yaml`:
   ```yaml
   logger:
     default: warning
     logs:
       custom_components.venice_ai: debug
   ```
2. Restart Home Assistant and reproduce the issue.
3. Capture the relevant log lines from `home-assistant.log` before opening an issue.

### Getting help

When opening a bug report, please include:

* Home Assistant version (`Settings → About`)
* Integration version (visible in HACS under the Venice AI integration)
* Relevant log lines with debug logging enabled
* A short description of what you expected vs. what happened

## Support
If you encounter any issues or have feature requests, please open an issue on our [GitHub Issues page](https://github.com/grasponcrypto/venice_ai/issues).

## License
This project is licensed under the MIT License - see the LICENSE file for details.
