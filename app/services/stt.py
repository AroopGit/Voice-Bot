from faster_whisper import WhisperModel
import asyncio
import os
import numpy as np

class FasterWhisperSTT:
    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):
        """
        Initialize Faster-Whisper model with multilingual support.
        Args:
            model_size: Size of the model (tiny, base, small, medium, large-v2, etc.)
                       Use 'medium' or 'large' for better Hindi accuracy
            device: "cuda" or "cpu"
            compute_type: "float16", "int8_float16", "int8", etc.
        """
        print(f"Loading Faster-Whisper model: {model_size} on {device}...")
        print("🌍 Multilingual support enabled (Hindi, English, and more)")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Faster-Whisper model loaded.")


    async def transcribe(self, audio_data):
        """
        Transcribe audio data (bytes or numpy array) with enhanced accuracy.
        """
        if isinstance(audio_data, bytes):
            # Convert 16kHz mono 16-bit PCM to float32 normalized audio
            audio_np = np.frombuffer(audio_data, dtype=np.int16).flatten().astype(np.float32) / 32768.0
        else:
            audio_np = audio_data
        
        # Ensure audio is not empty
        if len(audio_np) == 0:
            return ""
        
        # Enhanced transcription with better parameters for accuracy
        segments, info = self.model.transcribe(
            audio_np,
            beam_size=5,              # Higher beam size for better accuracy
            best_of=5,                # Consider top 5 candidates
            temperature=0.0,          # Deterministic output
            vad_filter=True,          # Voice Activity Detection to filter silence
            vad_parameters=dict(
                threshold=0.5,        # VAD threshold
                min_speech_duration_ms=250,  # Minimum speech duration
                min_silence_duration_ms=500  # Minimum silence to split
            ),
            language=None,            # Auto-detect language
            condition_on_previous_text=True,  # Use context from previous segments
            word_timestamps=False,    # Disable for faster processing
            without_timestamps=True   # We only need text
        )
        
        # Collect all segments
        text = ""
        for segment in segments:
            text += segment.text
        
        # Clean up the text
        result = text.strip()
        
        # Log detected language
        if hasattr(info, 'language'):
            print(f"🌍 Detected language: {info.language} (confidence: {info.language_probability:.2f})")
        
        return result

