"""
Services module for the SwiftShip India Voice Bot Backend.
Contains all service components for STT, TTS, RAG, and language detection.
Optimized for Hindi, English, and Hinglish.
"""

from .language_detector import LanguageDetectorService, get_language_detector
from .stt_service import STTService, get_stt_service
from .tts_service import TTSService, get_tts_service
from .rag_service import RAGService, get_rag_service
from .streaming_service import StreamingService, get_streaming_service

__all__ = [
    "LanguageDetectorService",
    "get_language_detector",
    "STTService",
    "get_stt_service",
    "TTSService",
    "get_tts_service",
    "RAGService",
    "get_rag_service",
    "StreamingService",
    "get_streaming_service",
]
