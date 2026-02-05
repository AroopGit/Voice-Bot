"""
Vahak Logistics Voice Bot - Main FastAPI Application

High-performance multilingual RAG voice bot backend with:
- Real-time voice streaming via WebSocket
- Faster-Whisper STT
- Qdrant vector search
- Groq LLM integration
- Edge-TTS multilingual output
"""

import asyncio
import time
import uuid
import json
import io
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException, UploadFile, File, Form, Query,
    BackgroundTasks, Request
)
from fastapi.responses import (
    JSONResponse, StreamingResponse, Response,
    FileResponse, HTMLResponse
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/voicebot.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)

from config import settings, get_voice_for_language, get_language_name
from services import (
    StreamingService, get_streaming_service,
    RAGService, get_rag_service,
    STTService, get_stt_service,
    TTSService, get_tts_service,
    LanguageDetectorService, get_language_detector
)


# ============ Pydantic Models ============

class TextQueryRequest(BaseModel):
    """Request model for text-based queries."""
    text: str = Field(..., min_length=1, description="Query text")
    language: Optional[str] = Field(None, description="Language code (auto-detected if not provided)")
    generate_audio: bool = Field(True, description="Whether to generate TTS audio")
    use_fast_model: bool = Field(True, description="Use fast (8B) vs quality (70B) LLM model")


class TextQueryResponse(BaseModel):
    """Response model for text queries."""
    query: str
    response: str
    language: str
    detected_language: str
    documents_found: int
    has_audio: bool
    metrics: dict


class VoiceQueryRequest(BaseModel):
    """Request model for voice queries via REST API."""
    sample_rate: int = Field(16000, description="Audio sample rate")
    language: Optional[str] = Field(None, description="Force specific language")


class DocumentUploadRequest(BaseModel):
    """Request for uploading documents to knowledge base."""
    text: str = Field(..., min_length=1)
    source: str = Field("api_upload")
    metadata: Optional[dict] = None


class BatchDocumentRequest(BaseModel):
    """Request for batch document upload."""
    documents: List[DocumentUploadRequest]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: dict
    uptime_seconds: float


# ============ Application Lifecycle ============

# Track startup time
APP_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle handler."""
    logger.info("=" * 50)
    logger.info("🚀 Vahak Voice Bot Starting...")
    logger.info("=" * 50)
    
    # Initialize services
    streaming = get_streaming_service()
    success = await streaming.initialize()
    
    if success:
        logger.info("✅ All services initialized successfully")
        
        # Load initial knowledge base
        await load_initial_knowledge_base()
    else:
        logger.error("❌ Service initialization failed")
    
    yield
    
    # Cleanup
    logger.info("👋 Shutting down Voice Bot...")


# ============ FastAPI App ============

app = FastAPI(
    title="Vahak Voice Bot API",
    description="High-performance multilingual RAG voice bot for customer service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Helper Functions ============

async def load_initial_knowledge_base():
    """Load initial knowledge base documents on startup."""
    kb_path = Path(__file__).parent / "data" / "knowledge_base"
    
    if not kb_path.exists():
        logger.warning(f"Knowledge base directory not found: {kb_path}")
        return
    
    rag = get_rag_service()
    documents = []
    
    try:
        for file_path in kb_path.glob("*.txt"):
            try:
                content = file_path.read_text(encoding="utf-8")
                
                # Split into chunks if large
                chunks = rag.chunk_text(content) if len(content) > 1000 else [content]
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "source": file_path.stem,
                        "metadata": {
                            "file": file_path.name,
                            "chunk": i + 1
                        }
                    })
                    
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        if documents:
            await rag.add_documents_batch(documents)
            logger.info(f"📚 Loaded {len(documents)} documents into knowledge base")
        else:
            logger.warning("No documents found in knowledge base directory")
            
    except Exception as e:
        logger.error(f"Knowledge base loading failed: {e}")


# ============ API Endpoints ============

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns status of all services.
    """
    streaming = get_streaming_service()
    health = await streaming.health_check()
    
    return HealthResponse(
        status=health["status"],
        version="1.0.0",
        services=health.get("services", {}),
        uptime_seconds=time.time() - APP_START_TIME
    )


@app.get("/api/metrics")
async def get_metrics():
    """Get performance metrics for all services."""
    streaming = get_streaming_service()
    return streaming.get_metrics()


@app.post("/api/text-query")
async def text_query(request: TextQueryRequest):
    """
    Text-based RAG query endpoint.
    Use this for testing the RAG pipeline without voice.
    
    Returns:
        JSON response with answer and optional base64-encoded audio
    """
    start_time = time.perf_counter()
    
    streaming = get_streaming_service()
    
    try:
        result = await streaming.process_text_query(
            text=request.text,
            language=request.language,
            generate_audio=request.generate_audio
        )
        
        response = {
            "query": request.text,
            "response": result.get("response", ""),
            "language": result.get("language", "en"),
            "documents_found": len(result.get("documents", [])),
            "metrics": result.get("metrics", {}),
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 1)
        }
        
        if result.get("audio") and request.generate_audio:
            import base64
            response["audio_base64"] = base64.b64encode(result["audio"]).decode()
            response["audio_format"] = "mp3"
        
        return response
        
    except Exception as e:
        logger.error(f"Text query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice-query")
async def voice_query(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, or raw PCM)"),
    sample_rate: int = Form(16000),
    language: Optional[str] = Form(None)
):
    """
    Voice query endpoint - audio in, audio out.
    Upload an audio file and get the response as audio.
    
    Args:
        audio: Audio file to process
        sample_rate: Sample rate of audio
        language: Optional language code to force
        
    Returns:
        Audio response in MP3 format
    """
    start_time = time.perf_counter()
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        # Get services
        streaming = get_streaming_service()
        stt = get_stt_service()
        rag = get_rag_service()
        tts = get_tts_service()
        lang_detector = get_language_detector()
        
        # Step 1: Transcribe
        transcript, detected_lang, confidence = await stt.transcribe(
            audio_data, sample_rate, language
        )
        
        if not transcript:
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Step 2: Refine language detection
        if not language and confidence < 0.8:
            detected_lang, _ = lang_detector.detect(transcript)
        
        final_lang = language or detected_lang
        
        # Step 3: RAG query
        response_text, documents = await rag.query(
            query=transcript,
            language=final_lang,
            stream=False
        )
        
        # Step 4: Generate TTS
        audio_response = await tts.synthesize(response_text, final_lang)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Voice query complete: '{transcript[:50]}...' -> '{response_text[:50]}...' "
            f"(lang={final_lang}, time={total_time:.0f}ms)"
        )
        
        # Return audio with metadata headers
        return Response(
            content=audio_response,
            media_type="audio/mpeg",
            headers={
                "X-Transcript": transcript[:200],
                "X-Response": response_text[:200],
                "X-Language": final_lang,
                "X-Processing-Time-Ms": str(round(total_time, 1))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-knowledge")
async def upload_knowledge(request: DocumentUploadRequest):
    """
    Upload a single document to the knowledge base.
    """
    rag = get_rag_service()
    
    try:
        doc_id = await rag.add_document(
            text=request.text,
            source=request.source,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "document_id": doc_id,
            "message": "Document added to knowledge base"
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-knowledge/batch")
async def upload_knowledge_batch(request: BatchDocumentRequest):
    """
    Upload multiple documents to the knowledge base in batch.
    """
    rag = get_rag_service()
    
    try:
        documents = [
            {
                "text": doc.text,
                "source": doc.source,
                "metadata": doc.metadata or {}
            }
            for doc in request.documents
        ]
        
        doc_ids = await rag.add_documents_batch(documents)
        
        return {
            "status": "success",
            "document_count": len(doc_ids),
            "document_ids": doc_ids
        }
        
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    source: str = Form("file_upload")
):
    """
    Upload a text file to the knowledge base.
    """
    rag = get_rag_service()
    
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        # Chunk the content
        chunks = rag.chunk_text(text)
        
        documents = [
            {
                "text": chunk,
                "source": source,
                "metadata": {
                    "filename": file.filename,
                    "chunk": i + 1
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
        doc_ids = await rag.add_documents_batch(documents)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(doc_ids),
            "document_ids": doc_ids
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search_knowledge(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
    threshold: float = Query(0.5, ge=0, le=1)
):
    """
    Search the knowledge base directly.
    """
    rag = get_rag_service()
    
    try:
        results = await rag.retrieve(query=q, top_k=top_k, threshold=threshold)
        
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tts")
async def text_to_speech(
    text: str = Query(..., min_length=1, max_length=1000),
    language: str = Query("en")
):
    """
    Generate speech from text.
    """
    tts = get_tts_service()
    
    try:
        audio = await tts.synthesize(text, language)
        
        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voices")
async def list_available_voices(
    language: Optional[str] = Query(None, description="Filter by language")
):
    """
    List available TTS voices.
    """
    tts = get_tts_service()
    
    try:
        voices = await tts.list_voices(language)
        
        return {
            "voices": [
                {
                    "name": v.get("ShortName", ""),
                    "locale": v.get("Locale", ""),
                    "gender": v.get("Gender", "")
                }
                for v in voices[:50]  # Limit response size
            ],
            "count": len(voices)
        }
        
    except Exception as e:
        logger.error(f"Voice listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/languages")
async def list_supported_languages():
    """
    List supported languages.
    """
    from config import LANGUAGE_VOICE_MAP, LANGUAGE_NAMES
    
    languages = []
    for code, voice in LANGUAGE_VOICE_MAP.items():
        languages.append({
            "code": code,
            "name": LANGUAGE_NAMES.get(code, code),
            "voice": voice
        })
    
    return {"languages": languages, "count": len(languages)}


# ============ WebSocket Endpoint ============

@app.websocket("/ws")
async def websocket_voice_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Protocol:
    - Client sends: binary audio chunks (16-bit PCM, 16kHz)
    - Server sends: 
        - JSON: {"type": "status", "message": "..."}
        - JSON: {"type": "user_transcript", "text": "..."}
        - JSON: {"type": "bot_response", "text": "..."}
        - Binary: MP3 audio response
        - JSON: {"type": "error", "message": "..."}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    logger.info(f"WebSocket connected: {session_id}")
    
    streaming = get_streaming_service()
    
    if not streaming.is_initialized:
        await streaming.initialize()
    
    try:
        await streaming.handle_websocket(websocket, session_id)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        streaming.remove_session(session_id)


@app.websocket("/ws/voice-stream")
async def websocket_voice_stream_alt(websocket: WebSocket):
    """Alternative WebSocket endpoint (alias)."""
    await websocket_voice_stream(websocket)


# ============ Static Files ============

# ============ Static Files ============

# Get root directory of the project
root_dir = Path(__file__).parent.parent

@app.get("/voice-app.js")
async def serve_voice_js():
    """Serve the voice assistant JavaScript file from root"""
    return FileResponse(root_dir / "voice-app.js")

@app.get("/AIEnhancer_oneture_logo-removebg-preview.png")
async def serve_logo1():
    """Serve the Oneture logo from root"""
    return FileResponse(root_dir / "AIEnhancer_oneture_logo-removebg-preview.png")

@app.get("/aws_logo.png")
async def serve_logo2():
    """Serve the AWS logo from root"""
    return FileResponse(root_dir / "aws_logo.png")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main UI page"""
    index_path = root_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("index.html not found in root directory", status_code=404)

# Keep /static for other assets or legacy support
app.mount("/static", StaticFiles(directory=str(root_dir)), name="static")


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
