from faster_whisper import WhisperModel
import asyncio
import os
import numpy as np

class FasterWhisperSTT:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        """
        Initialize Faster-Whisper model with multilingual support.
        Optimized for Hindi, English, and Marathi accuracy.
        Args:
            model_size: Size of the model (tiny, base, small, medium, large-v2, etc.)
                       Use 'medium' or 'large' for better Hindi/Marathi accuracy
            device: "cuda" or "cpu"
            compute_type: "float16", "int8_float16", "int8", etc.
        """
        print(f"Loading Faster-Whisper model: {model_size} on {device}...")
        print("🌍 Multilingual support enabled (Hindi, English, Marathi)")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.detected_language = "en"
        self.language_confidence = 0.0
        print("Faster-Whisper model loaded.")

    async def transcribe(self, audio_data):
        """
        Transcribe audio data with HIGH ACCURACY for Hindi/Marathi/English.
        Prioritizes accuracy over speed.
        """
        try:
            if isinstance(audio_data, bytes):
                # Convert 16kHz mono 16-bit PCM to float32 normalized audio
                audio_np = np.frombuffer(audio_data, dtype=np.int16).flatten().astype(np.float32) / 32768.0
            else:
                audio_np = audio_data
            
            # Ensure audio is not empty
            if len(audio_np) == 0:
                return ""
            
            # Initial prompt with Hindi/Marathi/English context
            initial_prompt = (
                "यह एक logistics कॉल है। Vahak truck booking service। "
                "मुंबई से दिल्ली तक ट्रक बुक करो। "
                "I wanted to book a truck। "
                "मराठी मध्ये बोला। "
                "हिंदी, मराठी, English में बात।"
            )
            
            # HIGH ACCURACY transcription settings
            segments, info = self.model.transcribe(
                audio_np,
                
                # Language settings - let model auto-detect for best accuracy
                task="transcribe",
                language=None,  # Auto-detect language for better accuracy
                
                # Higher beam size for better accuracy
                beam_size=5,
                best_of=5,
                
                # Temperature settings for accuracy
                temperature=0.0,  # Greedy decoding for consistency
                
                # Context conditioning - helps with Hindi/Marathi
                condition_on_previous_text=True,
                
                # Initial prompt to prime the model
                initial_prompt=initial_prompt,
                
                # VAD filter for clean audio processing
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.4,
                    min_speech_duration_ms=200,
                    min_silence_duration_ms=300,
                    speech_pad_ms=100,
                ),
                
                # Suppress blank/noise tokens
                suppress_blank=True,
            )
            
            # Collect all segments
            text = ""
            for segment in segments:
                text += segment.text
            
            # Clean up the text
            result = text.strip()
            
            # Store detected language info
            if hasattr(info, 'language'):
                self.detected_language = info.language
                self.language_confidence = info.language_probability
                print(f"🌍 Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            
            return result
            
        except Exception as e:
            print(f"❌ STT Error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def get_detected_language(self):
        """Return the last detected language code."""
        return self.detected_language
    
    def get_language_confidence(self):
        """Return confidence of language detection."""
        return self.language_confidence
