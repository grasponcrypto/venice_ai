"""Constants for the Venice AI Conversation integration."""

from datetime import timedelta

DOMAIN = "venice_ai"

# Coordinator refresh interval — must be a timedelta for DataUpdateCoordinator
UPDATE_INTERVAL = timedelta(hours=12)

CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "e2ee-gemma-4-31b"  # Venice AI default model with function calling support
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 512
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0

# Venice AI reasoning model options
CONF_STRIP_THINKING_RESPONSE = "strip_thinking_response"
RECOMMENDED_STRIP_THINKING_RESPONSE = True
CONF_DISABLE_THINKING = "disable_thinking"
# Disable thinking by default for automations to reduce latency, token usage, and cost,
# as complex reasoning is rarely needed for standard Home Assistant actions.
RECOMMENDED_DISABLE_THINKING = True

# MED-3: Opt-in streaming for conversation responses. When enabled, the
# conversation entity consumes the Venice AI streaming chat API via the
# VeniceConversationService and accumulates deltas (including tool calls).
CONF_STREAM_RESPONSE = "stream_response"
RECOMMENDED_STREAM_RESPONSE = True

# Venice AI TTS options
CONF_TTS_MODEL = "tts_model"
RECOMMENDED_TTS_MODEL = "tts-kokoro"
CONF_TTS_VOICE = "tts_voice"
RECOMMENDED_TTS_VOICE = "bm_daniel"
CONF_TTS_RESPONSE_FORMAT = "tts_response_format"
RECOMMENDED_TTS_RESPONSE_FORMAT = "mp3"
CONF_TTS_SPEED = "tts_speed"
RECOMMENDED_TTS_SPEED = 1.0

# Venice AI STT options
CONF_STT_MODEL = "stt_model"
RECOMMENDED_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
CONF_STT_RESPONSE_FORMAT = "stt_response_format"
RECOMMENDED_STT_RESPONSE_FORMAT = "json"
CONF_STT_TIMESTAMPS = "stt_timestamps"
RECOMMENDED_STT_TIMESTAMPS = False

# Conversation tool iteration limit
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
RECOMMENDED_MAX_TOOL_ITERATIONS = 5
MAX_CHAT_LOG_LENGTH = 50

# Maximum audio buffer size for STT to prevent memory spikes (10 MB).
# Venice AI does not support chunked/streaming STT uploads; the entire audio
# payload must be buffered before submission. Recordings exceeding this limit
# are rejected early with an ERROR result rather than causing an OOM spike.
MAX_STT_BUFFER_SIZE = 10 * 1024 * 1024

# Request timeouts are per operation, not one user setting. A stream can
# sit open for minutes while tokens arrive; a model-list call should fail
# in seconds. See DEFAULT_HTTP_TIMEOUT and the per-call constants below.

# Retry configuration constants (MED-4).
# Extracted from client.py so they can be tuned without touching client logic.
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0

# Feature minimum HA versions (MAINT-3).
# Reference table for conditional feature activation and user-facing docs.
FEATURE_MIN_VERSIONS: dict[str, str] = {
    "ai_task": "2024.8.0",
    "streaming_tts": "2024.4.0",
    "conversation_entity": "2023.10.0",
    "sensor_total_increasing": "2021.12.0",
}

# QUAL-2 / PERF-4: httpx connection-pool and per-request timeout defaults.
# These are intentionally hardcoded. Chat, streaming chat, TTS, STT, and
# image generation wait for different lengths of time; one options slider
# cannot cover all of them without cutting off a stream or hanging a
# short call. client.py passes the matching constant as the per-request
# timeout. DEFAULT_HTTP_TIMEOUT is only the client-wide fallback.
DEFAULT_HTTP_TIMEOUT = 30.0
DEFAULT_HTTP_KEEPALIVE = 5
DEFAULT_HTTP_MAX_CONNECTIONS = 10
DEFAULT_CHAT_TIMEOUT = 120.0
DEFAULT_CHAT_STREAM_TIMEOUT = 300.0
DEFAULT_TTS_TIMEOUT = 60.0
DEFAULT_STT_TIMEOUT = 60.0
DEFAULT_IMAGE_TIMEOUT = 120.0
