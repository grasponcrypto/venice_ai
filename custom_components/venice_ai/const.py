"""Constants for the Venice AI Conversation integration."""

from datetime import timedelta

DOMAIN = "venice_ai"

# Coordinator refresh interval — must be a timedelta for DataUpdateCoordinator
UPDATE_INTERVAL = timedelta(hours=1)

# Centralized voluptuous_openapi detection
try:
    from voluptuous_openapi import convert as voluptuous_convert  # noqa: F401
    HAS_VOLUPTUOUS_OPENAPI = True
except ImportError:
    HAS_VOLUPTUOUS_OPENAPI = False

CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "llama-3.3-70b"  # Venice AI default model with function calling support
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 512
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0

# Venice AI reasoning model options
CONF_STRIP_THINKING_RESPONSE = "strip_thinking_response"

# Venice AI TTS options
CONF_TTS_MODEL = "tts_model"
RECOMMENDED_TTS_MODEL = "tts-kokoro"
CONF_TTS_VOICE = "tts_voice"
RECOMMENDED_TTS_VOICE = "bm_daniel"
CONF_TTS_RESPONSE_FORMAT = "tts_response_format"
RECOMMENDED_TTS_RESPONSE_FORMAT = "mp3"
CONF_TTS_SPEED = "tts_speed"
RECOMMENDED_TTS_SPEED = 1.0

# Venice AI TTS voices (shared between config flow and TTS provider voice list)
VENICE_TTS_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "zf_xiaobei",
    "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
    "zm_yunxia", "zm_yunyang", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega",
    "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune",
    "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa",
    "ef_dora", "em_alex", "em_santa",
]

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

# Maximum number of concurrent conversations held in-memory (LRU eviction)
MAX_CHAT_HISTORY_SIZE = 20

# Maximum audio buffer size for STT to prevent memory spikes (10 MB)
MAX_STT_BUFFER_SIZE = 10 * 1024 * 1024
