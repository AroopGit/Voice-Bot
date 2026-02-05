"""
Model Download Script for Voice Bot
Downloads all required models for the voice assistant
"""

import os
import sys
from pathlib import Path

def download_mistral_model():
    """
    Download Mistral 7B Instruct GGUF model
    """
    print("=" * 60)
    print("📥 MISTRAL 7B MODEL DOWNLOAD")
    print("=" * 60)
    print()
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    print("⚠️  IMPORTANT: Mistral 7B model is ~4GB")
    print()
    print("Download options:")
    print()
    print("Option 1: Hugging Face (Recommended)")
    print("  URL: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    print("  File: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    print()
    print("Option 2: Use smaller model for testing")
    print("  URL: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    print("  File: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (~700MB)")
    print()
    
    choice = input("Choose option (1/2) or 's' to skip: ").strip()
    
    if choice == "1":
        print()
        print("📋 Manual Download Instructions:")
        print("1. Go to: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        print("2. Click on 'Files and versions'")
        print("3. Download: mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        print(f"4. Save to: {model_dir.absolute()}")
        print()
        input("Press Enter when download is complete...")
        
    elif choice == "2":
        print()
        print("📋 Manual Download Instructions (Smaller Model):")
        print("1. Go to: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        print("2. Click on 'Files and versions'")
        print("3. Download: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        print(f"4. Save to: {model_dir.absolute()}")
        print()
        print("⚠️  Note: Update main_websocket.py to use this model path")
        print()
        input("Press Enter when download is complete...")
    else:
        print("⏭️  Skipping model download")
        print("   System will use fallback responses without LLM")

def check_faster_whisper():
    """
    Check if Faster-Whisper models are available
    """
    print()
    print("=" * 60)
    print("🎤 FASTER-WHISPER STT")
    print("=" * 60)
    print()
    
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper package installed")
        print()
        print("📥 Downloading Whisper 'small' model (first run only)...")
        print("   This may take a few minutes...")
        
        # This will download the model if not present
        model = WhisperModel("small", device="cpu", compute_type="int8")
        print("✅ Whisper model ready!")
        
    except ImportError:
        print("❌ faster-whisper not installed")
        print("   Run: pip install faster-whisper")
    except Exception as e:
        print(f"⚠️  Error loading Whisper model: {e}")

def check_sentence_transformers():
    """
    Check if Sentence-Transformers models are available
    """
    print()
    print("=" * 60)
    print("🔍 SENTENCE-TRANSFORMERS (RAG Embeddings)")
    print("=" * 60)
    print()
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers package installed")
        print()
        print("📥 Downloading embedding model 'all-MiniLM-L6-v2'...")
        print("   This may take a few minutes...")
        
        # This will download the model if not present
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model ready!")
        
    except ImportError:
        print("❌ sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
    except Exception as e:
        print(f"⚠️  Error loading embedding model: {e}")

def check_edge_tts():
    """
    Check if Edge-TTS is available
    """
    print()
    print("=" * 60)
    print("🔊 EDGE-TTS")
    print("=" * 60)
    print()
    
    try:
        import edge_tts
        print("✅ edge-tts package installed")
        print("✅ No model download needed (cloud-based)")
        
    except ImportError:
        print("❌ edge-tts not installed")
        print("   Run: pip install edge-tts")

def main():
    print()
    print("🤖 VOICE BOT MODEL SETUP")
    print("=" * 60)
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    print()
    
    # Download/check each component
    download_mistral_model()
    check_faster_whisper()
    check_sentence_transformers()
    check_edge_tts()
    
    # Summary
    print()
    print("=" * 60)
    print("✅ MODEL SETUP COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run verification: .\\verify_system.ps1")
    print("2. Start server: python main_websocket.py")
    print()

if __name__ == "__main__":
    main()
