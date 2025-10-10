"""Constants for the Venice AI Conversation integration."""

import logging

DOMAIN = "venice_ai"
LOGGER = logging.getLogger(__package__)

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "llama-3.3-70b"  # Venice AI default model with function calling support
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0

# Venice AI doesn't use reasoning_effort, but we'll keep the config option
# for compatibility
CONF_REASONING_EFFORT = "reasoning_effort"
RECOMMENDED_REASONING_EFFORT = "low"

# Venice AI reasoning model options
CONF_STRIP_THINKING_RESPONSE = "strip_thinking_response"
CONF_DISABLE_THINKING = "disable_thinking"

# Venice AI TTS options
CONF_TTS_MODEL = "tts_model"
RECOMMENDED_TTS_MODEL = "tts-kokoro"
CONF_TTS_VOICE = "tts_voice"
RECOMMENDED_TTS_VOICE = "bm_daniel"
CONF_TTS_RESPONSE_FORMAT = "tts_response_format"
RECOMMENDED_TTS_RESPONSE_FORMAT = "mp3"
CONF_TTS_SPEED = "tts_speed"
RECOMMENDED_TTS_SPEED = 1.0

# Venice AI doesn't have unsupported models list currently
UNSUPPORTED_MODELS = []
