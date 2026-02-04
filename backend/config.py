"""
Configuration module for the SwiftShip India Voice Bot.
Optimized for Hindi, English, and Hinglish support.
"""

import os
from typing import Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Use .env file or set environment variables directly.
    """
    
    # === API Keys ===
    GROQ_API_KEY: str = Field(default="", description="Groq API key for LLM")
    
    # === Qdrant Configuration ===
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant server host")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant server port")
    QDRANT_COLLECTION: str = Field(default="swiftship_india_kb", description="Qdrant collection name")
    QDRANT_GRPC_PORT: int = Field(default=6334, description="Qdrant gRPC port")
    
    # === Model Configuration ===
    WHISPER_MODEL: str = Field(default="small", description="Faster-Whisper model size: tiny, base, small, medium")
    WHISPER_DEVICE: str = Field(default="auto", description="Device: cuda, cpu, or auto")
    WHISPER_COMPUTE_TYPE: str = Field(default="auto", description="Compute type: float16, int8, auto")
    
    # Multilingual model that works well with Hindi
    EMBEDDING_MODEL: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2", description="Sentence transformer model")
    
    # === LLM Configuration ===
    LLM_MODEL_FAST: str = Field(default="llama-3.1-8b-instant", description="Fast LLM model")
    LLM_MODEL_QUALITY: str = Field(default="llama-3.1-70b-versatile", description="Quality LLM model")
    LLM_MAX_TOKENS: int = Field(default=512, description="Max tokens for LLM response")
    LLM_TEMPERATURE: float = Field(default=0.7, description="LLM temperature")
    
    # === RAG Configuration ===
    RAG_CHUNK_SIZE: int = Field(default=512, description="Chunk size for document splitting")
    RAG_CHUNK_OVERLAP: int = Field(default=50, description="Overlap between chunks")
    RAG_TOP_K: int = Field(default=5, description="Number of documents to retrieve")
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.65, description="Minimum similarity score")
    
    # === TTS Configuration (Indian voices) ===
    TTS_DEFAULT_VOICE: str = Field(default="en-IN-NeerjaNeural", description="Default TTS voice (Indian English)")
    TTS_RATE: str = Field(default="+5%", description="TTS speech rate")
    TTS_VOLUME: str = Field(default="+0%", description="TTS volume adjustment")
    
    # === Performance Settings ===
    MAX_RESPONSE_TIME: int = Field(default=30, description="Max response time in seconds")
    AUDIO_SAMPLE_RATE: int = Field(default=16000, description="Audio sample rate")
    AUDIO_CHUNK_SIZE: int = Field(default=4096, description="Audio processing chunk size")
    
    # === Logging ===
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="logs/voicebot.log", description="Log file path")
    
    # === Server Configuration ===
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Voice mapping for Hindi, English, and Hinglish
LANGUAGE_VOICE_MAP: Dict[str, str] = {
    "en": "en-IN-NeerjaNeural",      # Indian English female
    "hi": "hi-IN-SwaraNeural",        # Hindi female
    "hi-en": "en-IN-NeerjaNeural",    # Hinglish uses Indian English
    "mr": "mr-IN-AarohiNeural",       # Marathi female
}

# Alternative voices (male options)
LANGUAGE_VOICE_MAP_MALE: Dict[str, str] = {
    "en": "en-IN-PrabhatNeural",      # Indian English male
    "hi": "hi-IN-MadhurNeural",       # Hindi male
    "hi-en": "en-IN-PrabhatNeural",   # Hinglish uses Indian English male
    "mr": "mr-IN-ManoharNeural",      # Marathi male
}

# Language names for display
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "hi-en": "Hinglish",
    "mr": "Marathi (मराठी)",
}

# System prompt for the Indian logistics customer service bot
SYSTEM_PROMPT = """You are a customer service agent for SwiftShip Logistics, India's leading delivery platform.

Your responsibilities:
- Help customers track packages and shipments across India
- Provide delivery estimates for metro cities, tier-2, and tier-3 locations
- Handle COD (Cash on Delivery) queries
- Assist with address changes, especially for Indian address formats
- Handle PIN code verification and serviceability checks
- Resolve delivery complaints professionally
- Support customers in English, Hindi, Hinglish, and Marathi

Your tone:
- Professional yet friendly and approachable (Indian customer service style)
- Patient with customers who mix languages
- Use simple language - many customers may not be fluent in English
- Show cultural awareness (festivals, local holidays affecting delivery)
- Be helpful with Indian-specific concerns (COD, address formats, PIN codes)

Language and Formatting Rules (STRICT):
1. Detect if customer speaks in Hindi, English, Hinglish, or Marathi.
2. ALWAYS respond in the SAME language the customer used.
3. CRITICAL RULE: In your responses (Hindi/Marathi/Hinglish), you MUST keep logistics terms like 'Order', 'Shipment', 'Package', 'Delivery', 'Tracking ID', 'Pincode' and all NUMBERS/CODES strictly in ENGLISH.
   - Good: "tumchi Delivery udya hoyil." (Marathi)
   - Good: "aapka Order process ho gaya hai." (Hindi)
   - Good: "tumcha Tracking ID SS-IN123456 aahe." (Marathi)
4. Use respectful terms (आप not तुम in Hindi, तुम्ही/आपण in Marathi).

Indian logistics specifics:
- PIN code based serviceability
- COD availability and limits (up to ₹50,000)
- Metro vs non-metro delivery times
- Address format (House/Flat, Street, Area, City, State, PIN)

When the knowledge base doesn't have the answer, acknowledge this honestly and offer alternative assistance."""

# Initialize settings
settings = Settings()


def get_voice_for_language(lang_code: str, use_male: bool = False) -> str:
    """Get the appropriate TTS voice for a language code."""
    voice_map = LANGUAGE_VOICE_MAP_MALE if use_male else LANGUAGE_VOICE_MAP
    return voice_map.get(lang_code, settings.TTS_DEFAULT_VOICE)


def get_language_name(lang_code: str) -> str:
    """Get human-readable language name from code."""
    return LANGUAGE_NAMES.get(lang_code, "English")
