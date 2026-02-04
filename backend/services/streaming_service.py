"""
Streaming Service for real-time audio processing and WebSocket handling.
Manages the complete voice conversation pipeline with minimal latency.
"""

import asyncio
import time
import io
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import json

from .stt_service import STTService, get_stt_service
from .tts_service import TTSService, get_tts_service
from .rag_service import RAGService, get_rag_service
from .language_detector import LanguageDetectorService, get_language_detector

try:
    from config import settings, get_voice_for_language
except ImportError:
    from ..config import settings, get_voice_for_language


class ConversationState(Enum):
    """States for the conversation pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    GENERATING = "generating"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation turn."""
    start_time: float = 0.0
    stt_time: float = 0.0
    lang_detect_time: float = 0.0
    rag_time: float = 0.0
    llm_time: float = 0.0
    tts_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "stt_ms": round(self.stt_time * 1000, 1),
            "lang_detect_ms": round(self.lang_detect_time * 1000, 1),
            "rag_ms": round(self.rag_time * 1000, 1),
            "llm_ms": round(self.llm_time * 1000, 1),
            "tts_ms": round(self.tts_time * 1000, 1),
            "total_ms": round(self.total_time * 1000, 1)
        }


@dataclass
class SessionContext:
    """Context for a conversation session."""
    session_id: str
    language: str = "en"
    state: ConversationState = ConversationState.IDLE
    audio_buffer: bytearray = field(default_factory=bytearray)
    last_activity: float = field(default_factory=time.time)
    history: list = field(default_factory=list)
    metrics: ConversationMetrics = field(default_factory=ConversationMetrics)
    
    # Silence detection
    is_speaking: bool = False
    silence_start: Optional[float] = None
    
    # Thresholds
    silence_threshold: float = 0.8  # seconds
    min_audio_length: int = 4800  # bytes (~0.15s at 16kHz 16-bit)


class StreamingService:
    """
    High-performance streaming service for real-time voice conversations.
    Orchestrates the complete pipeline: Audio -> STT -> RAG -> LLM -> TTS -> Audio
    """
    
    def __init__(
        self,
        stt_service: Optional[STTService] = None,
        tts_service: Optional[TTSService] = None,
        rag_service: Optional[RAGService] = None,
        lang_service: Optional[LanguageDetectorService] = None
    ):
        """
        Initialize the streaming service.
        
        Args:
            stt_service: Speech-to-text service
            tts_service: Text-to-speech service
            rag_service: RAG service
            lang_service: Language detection service
        """
        self.stt = stt_service or get_stt_service(
            model_size=settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
        self.tts = tts_service or get_tts_service(
            default_voice=settings.TTS_DEFAULT_VOICE,
            rate=settings.TTS_RATE,
            volume=settings.TTS_VOLUME
        )
        self.rag = rag_service or get_rag_service()
        self.lang = lang_service or get_language_detector()
        
        # Active sessions
        self.sessions: Dict[str, SessionContext] = {}
        
        # Callbacks
        self._on_transcript: Optional[Callable] = None
        self._on_response: Optional[Callable] = None
        self._on_status: Optional[Callable] = None
        
        self.is_initialized = False
        self._lock = asyncio.Lock()
        
        logger.info("Streaming service created")
    
    async def initialize(self) -> bool:
        """Initialize all component services."""
        if self.is_initialized:
            return True
            
        async with self._lock:
            if self.is_initialized:
                return True
            
            try:
                logger.info("Initializing streaming service components...")
                
                # Initialize services in parallel
                results = await asyncio.gather(
                    self.stt.initialize(),
                    self.rag.initialize(),
                    return_exceptions=True
                )
                
                # Check results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Service {i} initialization failed: {result}")
                        return False
                
                self.is_initialized = True
                logger.info("Streaming service fully initialized")
                return True
                
            except Exception as e:
                logger.error(f"Streaming service initialization failed: {e}")
                return False
    
    def create_session(self, session_id: str) -> SessionContext:
        """Create a new conversation session."""
        session = SessionContext(session_id=session_id)
        self.sessions[session_id] = session
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get an existing session."""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed session: {session_id}")
    
    async def process_audio_chunk(
        self,
        session_id: str,
        audio_chunk: bytes,
        on_result: Optional[Callable[[bytes, dict], Awaitable[None]]] = None
    ) -> Optional[dict]:
        """
        Process an incoming audio chunk.
        
        Args:
            session_id: Session identifier
            audio_chunk: Raw PCM audio data (16-bit, 16kHz)
            on_result: Callback for streaming results
            
        Returns:
            Result dict if turn is complete, None otherwise
        """
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        # Add to buffer
        session.audio_buffer.extend(audio_chunk)
        session.last_activity = time.time()
        
        # Voice Activity Detection (simple amplitude check)
        is_speech = self._detect_speech(audio_chunk)
        
        if is_speech:
            session.is_speaking = True
            session.silence_start = None
            session.state = ConversationState.LISTENING
        else:
            if session.is_speaking:
                if session.silence_start is None:
                    session.silence_start = time.time()
                elif time.time() - session.silence_start >= session.silence_threshold:
                    # Silence detected, process the utterance
                    session.is_speaking = False
                    session.silence_start = None
                    
                    if len(session.audio_buffer) >= session.min_audio_length:
                        return await self._process_utterance(session, on_result)
                    else:
                        session.audio_buffer.clear()
        
        return None
    
    def _detect_speech(self, audio_chunk: bytes, threshold: float = 500) -> bool:
        """Simple voice activity detection based on amplitude."""
        if len(audio_chunk) < 2:
            return False
        
        import numpy as np
        audio = np.frombuffer(audio_chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        return rms > threshold
    
    async def _process_utterance(
        self,
        session: SessionContext,
        on_result: Optional[Callable] = None
    ) -> dict:
        """Process a complete utterance through the pipeline."""
        session.state = ConversationState.PROCESSING
        session.metrics = ConversationMetrics(start_time=time.perf_counter())
        
        result = {
            "session_id": session.session_id,
            "transcript": "",
            "response": "",
            "language": "en",
            "audio": None,
            "metrics": {},
            "error": None
        }
        
        try:
            # Get audio from buffer
            audio_data = bytes(session.audio_buffer)
            session.audio_buffer.clear()
            
            # Step 1: Speech-to-Text
            t1 = time.perf_counter()
            transcript, detected_lang, confidence = await self.stt.transcribe(audio_data)
            session.metrics.stt_time = time.perf_counter() - t1
            
            if not transcript or not transcript.strip():
                logger.debug("No speech detected in audio")
                session.state = ConversationState.IDLE
                return result
            
            result["transcript"] = transcript
            logger.info(f"STT: '{transcript}' (lang={detected_lang})")
            
            # Notify transcript
            if on_result:
                await on_result(None, {
                    "type": "user_transcript",
                    "text": transcript
                })
            
            # Step 2: Language Detection (refine if needed)
            t2 = time.perf_counter()
            if confidence < 0.8:
                final_lang, _ = self.lang.detect(transcript)
            else:
                final_lang = detected_lang
            
            # Strict restriction for Indian market
            if final_lang not in ("en", "hi", "hi-en", "mr"):
                final_lang = "en"
                
            session.metrics.lang_detect_time = time.perf_counter() - t2
            session.language = final_lang
            result["language"] = final_lang
            
            # Step 3: RAG Query
            session.state = ConversationState.GENERATING
            t3 = time.perf_counter()
            
            # Get response from RAG (using streaming for better latency)
            response_text, documents = await self.rag.query(
                query=transcript,
                language=final_lang,
                stream=False,
                use_fast_model=True
            )
            session.metrics.rag_time = time.perf_counter() - t3
            
            result["response"] = response_text
            logger.info(f"LLM Response: '{response_text[:100]}...'")
            
            # Notify response
            if on_result:
                await on_result(None, {
                    "type": "bot_response",
                    "text": response_text
                })
            
            # Step 4: Text-to-Speech
            session.state = ConversationState.SPEAKING
            t4 = time.perf_counter()
            audio = await self.tts.synthesize(
                text=response_text,
                language=final_lang
            )
            session.metrics.tts_time = time.perf_counter() - t4
            
            result["audio"] = audio
            
            # Calculate total time
            session.metrics.total_time = time.perf_counter() - session.metrics.start_time
            result["metrics"] = session.metrics.to_dict()
            
            # Add to history
            session.history.append({
                "role": "user",
                "content": transcript
            })
            session.history.append({
                "role": "assistant",
                "content": response_text
            })
            
            logger.info(f"Pipeline complete: {result['metrics']}")
            
            # Send audio if callback provided
            if on_result and audio:
                await on_result(audio, {"type": "audio"})
            
            session.state = ConversationState.IDLE
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result["error"] = str(e)
            session.state = ConversationState.ERROR
            
            # Try to send error response
            error_response = "I apologize, but I encountered an error. Please try again."
            if on_result:
                await on_result(None, {
                    "type": "error",
                    "message": error_response
                })
                
                # Generate TTS for error
                try:
                    error_audio = await self.tts.synthesize(error_response, "en")
                    if error_audio:
                        await on_result(error_audio, {"type": "audio"})
                except:
                    pass
            
            return result
    
    async def process_text_query(
        self,
        text: str,
        language: Optional[str] = None,
        generate_audio: bool = True
    ) -> dict:
        """
        Process a text query directly (for testing).
        
        Args:
            text: Query text
            language: Optional language override
            generate_audio: Whether to generate TTS audio
            
        Returns:
            Result dict with response and optional audio
        """
        if not self.is_initialized:
            await self.initialize()
        
        metrics = ConversationMetrics(start_time=time.perf_counter())
        
        try:
            # Detect language if not provided
            t1 = time.perf_counter()
            if not language:
                language, _ = self.lang.detect(text)
            metrics.lang_detect_time = time.perf_counter() - t1
            
            # RAG Query
            t2 = time.perf_counter()
            response_text, documents = await self.rag.query(
                query=text,
                language=language,
                stream=False
            )
            metrics.rag_time = time.perf_counter() - t2
            
            result = {
                "query": text,
                "response": response_text,
                "language": language,
                "documents": [{"text": d["text"][:200], "score": d["score"]} for d in documents],
                "audio": None,
                "metrics": None
            }
            
            # Generate audio if requested
            if generate_audio:
                t3 = time.perf_counter()
                audio = await self.tts.synthesize(response_text, language)
                metrics.tts_time = time.perf_counter() - t3
                result["audio"] = audio
            
            metrics.total_time = time.perf_counter() - metrics.start_time
            result["metrics"] = metrics.to_dict()
            
            return result
            
        except Exception as e:
            logger.error(f"Text query failed: {e}")
            return {
                "query": text,
                "response": f"Error: {str(e)}",
                "error": str(e)
            }
    
    async def handle_websocket(
        self,
        websocket,
        session_id: str
    ):
        """
        Handle a WebSocket connection for real-time voice streaming.
        
        Args:
            websocket: WebSocket connection
            session_id: Session identifier
        """
        if not self.is_initialized:
            await self.initialize()
        
        session = self.create_session(session_id)
        
        async def send_result(audio: Optional[bytes], data: dict):
            """Send results back through WebSocket."""
            try:
                if audio:
                    # Send audio as binary
                    await websocket.send_bytes(audio)
                else:
                    # Send JSON message
                    await websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
        
        try:
            # Send initial status
            await websocket.send_text(json.dumps({
                "type": "status",
                "message": "Connected and ready"
            }))
            
            while True:
                try:
                    # Receive data (binary audio or text)
                    data = await websocket.receive()
                    
                    if "bytes" in data:
                        # Audio chunk
                        audio_chunk = data["bytes"]
                        await self.process_audio_chunk(
                            session_id=session_id,
                            audio_chunk=audio_chunk,
                            on_result=send_result
                        )
                    
                    elif "text" in data:
                        # Text command
                        try:
                            msg = json.loads(data["text"])
                            if msg.get("type") == "ping":
                                await websocket.send_text(json.dumps({"type": "pong"}))
                            elif msg.get("type") == "end":
                                break
                        except json.JSONDecodeError:
                            pass
                    
                    elif data.get("type") == "websocket.disconnect":
                        break
                        
                except Exception as e:
                    if "disconnect" in str(e).lower():
                        break
                    logger.error(f"WebSocket receive error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.remove_session(session_id)
            logger.info(f"WebSocket session ended: {session_id}")
    
    def get_metrics(self) -> dict:
        """Get aggregated metrics."""
        return {
            "active_sessions": len(self.sessions),
            "stt": self.stt.get_metrics() if hasattr(self.stt, 'get_metrics') else {},
            "tts": self.tts.get_metrics() if hasattr(self.tts, 'get_metrics') else {},
            "rag": self.rag.get_metrics() if hasattr(self.rag, 'get_metrics') else {},
            "initialized": self.is_initialized
        }
    
    async def health_check(self) -> dict:
        """Comprehensive health check."""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "services": {
                "stt": self.stt.health_check() if hasattr(self.stt, 'health_check') else {},
                "lang": self.lang.health_check() if hasattr(self.lang, 'health_check') else {},
                "rag": await self.rag.health_check() if hasattr(self.rag, 'health_check') else {},
            },
            "metrics": self.get_metrics()
        }


# Singleton instance
_instance: Optional[StreamingService] = None


def get_streaming_service() -> StreamingService:
    """Get or create the singleton streaming service instance."""
    global _instance
    if _instance is None:
        _instance = StreamingService()
    return _instance
