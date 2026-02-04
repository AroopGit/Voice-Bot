"""
Speech-to-Text Service using Faster-Whisper.
Provides high-performance speech recognition with automatic language detection.
"""

import asyncio
import io
import time
import wave
from typing import Optional, Tuple, AsyncGenerator, List
from loguru import logger
import numpy as np

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.error("faster-whisper not installed")


class STTService:
    """
    High-performance Speech-to-Text service using Faster-Whisper.
    Optimized for real-time transcription with streaming support.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Optional[str] = None
    ):
        """
        Initialize the STT service.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2)
            device: Device to use (cuda, cpu, auto)
            compute_type: Compute type (float16, int8, auto)
            download_root: Directory to download models to
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.model: Optional[WhisperModel] = None
        self.is_initialized = False
        self._lock = asyncio.Lock()
        
        # Audio buffer for streaming
        self.audio_buffer = []
        self.sample_rate = 16000
        
        # Performance metrics
        self.total_transcriptions = 0
        self.total_time = 0.0
        
        logger.info(f"STT Service configured: model={model_size}, device={device}, compute={compute_type}")
    
    async def initialize(self) -> bool:
        """
        Load the Whisper model asynchronously.
        Call this at startup to preload the model.
        """
        if self.is_initialized:
            return True
            
        async with self._lock:
            if self.is_initialized:
                return True
                
            try:
                start_time = time.perf_counter()
                logger.info(f"Loading Faster-Whisper model: {self.model_size}...")
                
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None,
                    self._load_model
                )
                
                elapsed = time.perf_counter() - start_time
                logger.info(f"Whisper model loaded in {elapsed:.2f}s")
                self.is_initialized = True
                return True
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                return False
    
    def _load_model(self) -> WhisperModel:
        """Load the Whisper model (blocking call)."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper is not installed")
        
        # Determine device and compute type
        device = self.device
        compute_type = self.compute_type
        
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Initializing Whisper on {device} with {compute_type}")
        
        return WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=8 if device == "cpu" else 1,
            download_root=self.download_root
        )
    
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Sample rate of audio
            language: Optional language code to force
            
        Returns:
            Tuple of (transcribed_text, detected_language, confidence)
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not audio_data or len(audio_data) < 1000:
            logger.debug("Audio data too short for transcription")
            return "", "en", 0.0
        
        start_time = time.perf_counter()
        
        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_array(audio_data)
            
            if len(audio_array) < 1000:
                return "", "en", 0.0
            
            # Normalize audio
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self._transcribe_sync(audio_array, language)
            )
            
            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            transcribed_text = " ".join(text_parts)
            
            # Restrict to Hindi, English, and Marathi only
            detected_language = info.language if info else "en"
            if detected_language not in ("hi", "en", "mr"):
                detected_language = "en"
                
            confidence = info.language_probability if info else 0.0
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            self.total_transcriptions += 1
            self.total_time += elapsed
            
            logger.info(
                f"STT: '{transcribed_text[:50]}...' "
                f"(lang={detected_language}, conf={confidence:.2f}, time={elapsed:.0f}ms)"
            )
            
            return transcribed_text, detected_language, confidence
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", "en", 0.0
    
    def _transcribe_sync(self, audio_array: np.ndarray, language: Optional[str]):
        """Synchronous transcription (runs in thread pool)."""
        return self.model.transcribe(
            audio_array,
            language=language,
            task="transcribe",
            beam_size=1,  # Set to 1 for maximum speed (greedy search)
            best_of=1,
            temperature=0.0,
            vad_filter=True,  # Filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=250, # Shorter for faster detection
                speech_pad_ms=100,
            ),
            condition_on_previous_text=False,
            suppress_tokens=[], # Don't suppress common tokens for speed
            word_timestamps=False,
            without_timestamps=True,
        )
    
    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw PCM bytes to numpy array."""
        return np.frombuffer(audio_bytes, dtype=np.int16)
    
    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Stream transcription - transcribe audio chunks as they arrive.
        
        Args:
            audio_chunks: Async generator of audio byte chunks
            
        Yields:
            Tuple of (partial_text, detected_language)
        """
        if not self.is_initialized:
            await self.initialize()
        
        buffer = bytearray()
        min_chunk_size = self.sample_rate * 2  # ~1 second of audio
        
        async for chunk in audio_chunks:
            buffer.extend(chunk)
            
            # Process when we have enough audio
            if len(buffer) >= min_chunk_size:
                text, lang, _ = await self.transcribe(bytes(buffer))
                if text:
                    yield text, lang
                buffer.clear()
        
        # Process remaining buffer
        if len(buffer) > 0:
            text, lang, _ = await self.transcribe(bytes(buffer))
            if text:
                yield text, lang
    
    def add_to_buffer(self, chunk: bytes):
        """Add audio chunk to internal buffer."""
        self.audio_buffer.append(chunk)
    
    def get_buffered_audio(self) -> bytes:
        """Get all buffered audio and clear the buffer."""
        if not self.audio_buffer:
            return b""
        audio = b"".join(self.audio_buffer)
        self.audio_buffer.clear()
        return audio
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        self.audio_buffer.clear()
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        avg_time = self.total_time / max(1, self.total_transcriptions)
        return {
            "total_transcriptions": self.total_transcriptions,
            "total_time_ms": self.total_time,
            "average_time_ms": avg_time,
            "model": self.model_size,
            "initialized": self.is_initialized
        }
    
    def health_check(self) -> dict:
        """Check if the service is healthy."""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "model": self.model_size,
            "device": self.device,
            "metrics": self.get_metrics()
        }


# Singleton instance
_instance: Optional[STTService] = None


def get_stt_service(
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto"
) -> STTService:
    """Get or create the singleton STT service instance."""
    global _instance
    if _instance is None:
        _instance = STTService(model_size, device, compute_type)
    return _instance
