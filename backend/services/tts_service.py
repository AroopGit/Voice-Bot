"""
Text-to-Speech Service using Edge-TTS.
Optimized for Hindi, English, and Hinglish with Indian voices.
"""

import asyncio
import time
from typing import Optional, AsyncGenerator, Dict
from loguru import logger
import edge_tts

try:
    from config import LANGUAGE_VOICE_MAP, settings
except ImportError:
    from ..config import LANGUAGE_VOICE_MAP, settings


class TTSService:
    """
    High-performance Text-to-Speech service using Edge-TTS.
    Optimized for Indian market with Hindi, English, and Hinglish support.
    """
    
    # Indian voice configuration
    VOICE_MAP = {
        'en': 'en-IN-NeerjaNeural',      # Indian English female
        'hi': 'hi-IN-SwaraNeural',        # Hindi female
        'hi-en': 'en-IN-NeerjaNeural',    # Hinglish uses Indian English
    }
    
    # Voice cache for quick lookup
    _voice_cache: Dict[str, str] = {}
    
    def __init__(
        self,
        default_voice: str = "en-IN-NeerjaNeural",
        rate: str = "+5%",
        volume: str = "+0%"
    ):
        """
        Initialize the TTS service with Indian voices.
        
        Args:
            default_voice: Default voice (Indian English)
            rate: Speech rate adjustment
            volume: Volume adjustment
        """
        self.default_voice = default_voice
        self.rate = rate
        self.volume = volume
        
        # Preload Indian voices in cache
        self._voice_cache = self.VOICE_MAP.copy()
        
        # Performance metrics
        self.total_synthesized = 0
        self.total_time = 0.0
        self.total_chars = 0
        
        logger.info(f"TTS Service initialized for Indian market: default={default_voice}")
    
    async def synthesize(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None
    ) -> bytes:
        """
        Synthesize text to speech audio.
        
        Args:
            text: Text to synthesize (can be Hindi, English, or Hinglish)
            language: Language code ('en', 'hi', 'hi-en')
            voice: Optional specific voice to use
            
        Returns:
            Audio bytes in MP3 format
        """
        if not text or not text.strip():
            return b""
        
        start_time = time.perf_counter()
        
        # Select appropriate voice for language
        selected_voice = voice or self._get_voice_for_language(language)
        
        try:
            # Create TTS communicate object
            communicate = edge_tts.Communicate(
                text=text,
                voice=selected_voice,
                rate=self.rate,
                volume=self.volume
            )
            
            # Collect audio data
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            
            audio_data = b"".join(audio_chunks)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            self.total_synthesized += 1
            self.total_time += elapsed
            self.total_chars += len(text)
            
            logger.info(
                f"TTS: {len(audio_data)} bytes for {len(text)} chars "
                f"(lang={language}, voice={selected_voice.split('-')[-1]}, time={elapsed:.0f}ms)"
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return b""
    
    async def synthesize_stream(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks as they're generated.
        
        Args:
            text: Text to synthesize
            language: Language code
            voice: Optional specific voice
            
        Yields:
            Audio byte chunks as they're generated
        """
        if not text or not text.strip():
            return
        
        start_time = time.perf_counter()
        selected_voice = voice or self._get_voice_for_language(language)
        total_bytes = 0
        
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=selected_voice,
                rate=self.rate,
                volume=self.volume
            )
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    total_bytes += len(chunk["data"])
                    yield chunk["data"]
            
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"TTS stream: {total_bytes} bytes in {elapsed:.0f}ms"
            )
            
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
    
    def _get_voice_for_language(self, lang_code: str) -> str:
        """Get the appropriate Indian voice for a language code."""
        # Check cache first
        if lang_code in self._voice_cache:
            return self._voice_cache[lang_code]
        
        # Normalize language code
        if lang_code.startswith('hi'):
            if 'en' in lang_code or lang_code == 'hi-en':
                voice = self.VOICE_MAP['hi-en']
            else:
                voice = self.VOICE_MAP['hi']
        else:
            voice = self.VOICE_MAP.get('en', self.default_voice)
        
        # Cache and return
        self._voice_cache[lang_code] = voice
        return voice
    
    @staticmethod
    async def list_voices(language: Optional[str] = None) -> list:
        """
        List available TTS voices.
        
        Args:
            language: Optional language filter
            
        Returns:
            List of available voice information
        """
        try:
            voices = await edge_tts.list_voices()
            
            if language:
                # Filter for Indian voices
                if language in ('hi', 'hi-en', 'en'):
                    voices = [v for v in voices if 
                              v["Locale"].startswith("hi-IN") or 
                              v["Locale"].startswith("en-IN")]
                else:
                    voices = [v for v in voices if v["Locale"].lower().startswith(language)]
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []
    
    @staticmethod
    async def list_indian_voices() -> dict:
        """List all available Indian voices (Hindi and English)."""
        try:
            voices = await edge_tts.list_voices()
            
            indian_voices = {
                "hindi": [],
                "english_indian": []
            }
            
            for v in voices:
                if v["Locale"].startswith("hi-IN"):
                    indian_voices["hindi"].append({
                        "name": v.get("ShortName", ""),
                        "gender": v.get("Gender", ""),
                        "locale": v.get("Locale", "")
                    })
                elif v["Locale"].startswith("en-IN"):
                    indian_voices["english_indian"].append({
                        "name": v.get("ShortName", ""),
                        "gender": v.get("Gender", ""),
                        "locale": v.get("Locale", "")
                    })
            
            return indian_voices
            
        except Exception as e:
            logger.error(f"Failed to list Indian voices: {e}")
            return {"hindi": [], "english_indian": []}
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        avg_time = self.total_time / max(1, self.total_synthesized)
        chars_per_sec = self.total_chars / max(0.001, self.total_time / 1000)
        
        return {
            "total_synthesized": self.total_synthesized,
            "total_chars": self.total_chars,
            "total_time_ms": self.total_time,
            "average_time_ms": avg_time,
            "chars_per_second": chars_per_sec,
            "supported_languages": ["en", "hi", "hi-en"]
        }
    
    async def health_check(self) -> dict:
        """Check if the service is healthy with Indian voice test."""
        try:
            # Test with Hindi
            test_text = "नमस्ते, मैं आपकी मदद के लिए हूं।"
            start = time.perf_counter()
            audio = await self.synthesize(test_text, "hi")
            elapsed_hi = (time.perf_counter() - start) * 1000
            
            # Test with English
            test_text_en = "Hello, I am here to help you."
            start = time.perf_counter()
            audio_en = await self.synthesize(test_text_en, "en")
            elapsed_en = (time.perf_counter() - start) * 1000
            
            return {
                "status": "healthy" if audio and audio_en else "unhealthy",
                "hindi_voice": self.VOICE_MAP['hi'],
                "english_voice": self.VOICE_MAP['en'],
                "hinglish_voice": self.VOICE_MAP['hi-en'],
                "test_latency_hindi_ms": elapsed_hi,
                "test_latency_english_ms": elapsed_en,
                "metrics": self.get_metrics()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance
_instance: Optional[TTSService] = None


def get_tts_service(
    default_voice: str = "en-IN-NeerjaNeural",
    rate: str = "+5%",
    volume: str = "+0%"
) -> TTSService:
    """Get or create the singleton TTS service instance."""
    global _instance
    if _instance is None:
        _instance = TTSService(default_voice, rate, volume)
    return _instance
