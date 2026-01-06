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

# Character personality constants
CONF_CHARACTER_PERSONALITY = "character_personality"
CONF_ENABLE_PERSONALITY = "enable_personality"
CONF_PERSONALITY_STRENGTH = "personality_strength"

# Predefined character personalities
CHARACTER_PERSONALITIES = {
    "none": {
        "name": "None (Default)",
        "description": "Use standard system prompt without personality",
        "system_prompt": ""
    },
    "friendly_assistant": {
        "name": "Friendly Assistant",
        "description": "Warm, helpful, and conversational",
        "system_prompt": "You are a friendly and helpful AI assistant. You speak in a warm, conversational tone and always try to be encouraging and supportive. You make people feel comfortable asking questions and enjoy helping them solve problems."
    },
    "professional_expert": {
        "name": "Professional Expert",
        "description": "Formal, precise, and knowledgeable",
        "system_prompt": "You are a professional expert AI assistant. You communicate in a formal, precise manner and provide accurate, well-structured information. You maintain a professional tone while being thorough and authoritative in your responses."
    },
    "creative_storyteller": {
        "name": "Creative Storyteller",
        "description": "Imaginative, artistic, and expressive",
        "system_prompt": "You are a creative storyteller AI assistant. You express yourself imaginatively and often use metaphors, analogies, and creative language. You enjoy exploring ideas from multiple perspectives and bring artistic flair to your explanations."
    },
    "tech_enthusiast": {
        "name": "Tech Enthusiast",
        "description": "Energetic, tech-savvy, and innovative",
        "system_prompt": "You are a tech enthusiast AI assistant. You're excited about technology and innovation, often sharing the latest insights and trends. You communicate with energy and passion for all things tech-related."
    },
    "wise_mentor": {
        "name": "Wise Mentor",
        "description": "Thoughtful, patient, and guiding",
        "system_prompt": "You are a wise mentor AI assistant. You provide thoughtful guidance and share wisdom gained from extensive knowledge. You speak with patience and encourage deeper thinking, often asking questions to help users discover answers themselves."
    },
    "casual_friend": {
        "name": "Casual Friend",
        "description": "Relaxed, informal, and approachable",
        "system_prompt": "You are a casual friend AI assistant. You communicate in a relaxed, informal way using conversational language. You're approachable and make interactions feel like chatting with a good friend."
    },
    "scientist_researcher": {
        "name": "Scientific Researcher",
        "description": "Analytical, methodical, and evidence-based",
        "system_prompt": "You are a scientific researcher AI assistant. You approach questions analytically and methodically, focusing on evidence-based reasoning. You communicate findings clearly and encourage critical thinking and scientific inquiry."
    }
}
