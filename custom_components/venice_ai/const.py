"""Constants for the Venice AI Conversation integration."""

import logging

DOMAIN = "venice_ai"
LOGGER = logging.getLogger(__package__)

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "default"  # Venice AI uses "default" as their model name
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

# Venice AI doesn't have unsupported models list currently
UNSUPPORTED_MODELS = []
