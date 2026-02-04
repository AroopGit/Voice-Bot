"""
Setup script for the SwiftShip Voice Bot.
Downloads models, initializes services, and loads knowledge base.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


async def setup_models():
    """Download and initialize all ML models."""
    print("\n" + "=" * 60)
    print("SWIFTSHIP VOICE BOT - SETUP")
    print("=" * 60 + "\n")
    
    # Step 1: Check environment
    print("📋 Checking environment...")
    
    from config import settings
    
    checks = [
        ("GROQ_API_KEY", bool(settings.GROQ_API_KEY)),
        ("QDRANT_HOST", bool(settings.QDRANT_HOST)),
    ]
    
    for name, available in checks:
        status = "✅" if available else "⚠️ Not set"
        print(f"   {name}: {status}")
    
    if not settings.GROQ_API_KEY:
        print("\n⚠️  WARNING: GROQ_API_KEY is not set!")
        print("   The LLM service will not work without it.")
        print("   Please set it in your .env file or environment.\n")
    
    # Step 2: Initialize STT (downloads Whisper model)
    print("\n🎤 Setting up Speech-to-Text (Faster-Whisper)...")
    try:
        from services.stt_service import get_stt_service
        stt = get_stt_service(
            model_size=settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
        await stt.initialize()
        print(f"   ✅ Whisper model '{settings.WHISPER_MODEL}' loaded")
    except Exception as e:
        print(f"   ❌ Failed to load Whisper: {e}")
        return False
    
    # Step 3: Initialize Embeddings
    print("\n📚 Setting up Embeddings (Sentence-Transformers)...")
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(f"   ✅ Embedding model '{settings.EMBEDDING_MODEL}' loaded")
    except Exception as e:
        print(f"   ❌ Failed to load embeddings: {e}")
        return False
    
    # Step 4: Initialize Qdrant
    print("\n🔍 Setting up Vector Database (Qdrant)...")
    try:
        from services.rag_service import get_rag_service
        rag = get_rag_service()
        await rag.initialize()
        print(f"   ✅ Qdrant collection '{settings.QDRANT_COLLECTION}' ready")
    except Exception as e:
        print(f"   ❌ Failed to connect to Qdrant: {e}")
        print("   💡 Make sure Qdrant is running on localhost:6333")
        print("   💡 Start with: docker run -p 6333:6333 qdrant/qdrant")
        return False
    
    # Step 5: Load Knowledge Base
    print("\n📖 Loading Knowledge Base...")
    try:
        kb_path = Path(__file__).parent / "data" / "knowledge_base"
        
        if not kb_path.exists():
            print(f"   ⚠️ Knowledge base directory not found: {kb_path}")
            print("   Creating directory...")
            kb_path.mkdir(parents=True, exist_ok=True)
        
        files = list(kb_path.glob("*.txt"))
        
        if not files:
            print("   ⚠️ No knowledge base files found")
        else:
            documents = []
            for file_path in files:
                content = file_path.read_text(encoding="utf-8")
                chunks = rag.chunk_text(content) if len(content) > 1000 else [content]
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "source": file_path.stem,
                        "metadata": {"file": file_path.name, "chunk": i + 1}
                    })
            
            if documents:
                doc_ids = await rag.add_documents_batch(documents)
                print(f"   ✅ Loaded {len(doc_ids)} document chunks from {len(files)} files")
            
    except Exception as e:
        print(f"   ⚠️ Knowledge base loading error: {e}")
    
    # Step 6: Test TTS
    print("\n🔊 Testing Text-to-Speech (Edge-TTS)...")
    try:
        from services.tts_service import get_tts_service
        tts = get_tts_service()
        audio = await tts.synthesize("Test successful.", "en")
        if len(audio) > 100:
            print("   ✅ Edge-TTS working correctly")
        else:
            print("   ⚠️ TTS produced minimal output")
    except Exception as e:
        print(f"   ❌ TTS error: {e}")
    
    # Step 7: Test Groq (if API key available)
    if settings.GROQ_API_KEY:
        print("\n🤖 Testing LLM (Groq)...")
        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            response = client.chat.completions.create(
                model=settings.LLM_MODEL_FAST,
                messages=[{"role": "user", "content": "Say 'ready'"}],
                max_tokens=10
            )
            print(f"   ✅ Groq API connected ({settings.LLM_MODEL_FAST})")
        except Exception as e:
            print(f"   ❌ Groq error: {e}")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nTo start the server, run:")
    print("   python main.py")
    print("\nOr with uvicorn:")
    print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("=" * 60 + "\n")
    
    return True


def check_qdrant_running():
    """Check if Qdrant is running."""
    import socket
    from config import settings
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    
    try:
        result = sock.connect_ex((settings.QDRANT_HOST, settings.QDRANT_PORT))
        return result == 0
    except:
        return False
    finally:
        sock.close()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="WARNING")
    
    # Check if Qdrant is running
    if not check_qdrant_running():
        print("\n⚠️  Qdrant does not appear to be running!")
        print("   Please start Qdrant before running setup:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
        print("\n   Or install and run locally:")
        print("   https://qdrant.tech/documentation/quick-start/")
        
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run setup
    success = asyncio.run(setup_models())
    sys.exit(0 if success else 1)
