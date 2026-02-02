import edge_tts
import asyncio
import tempfile
import os

class EdgeTTSService:
    def __init__(self, voice="en-US-ChristopherNeural"):
        """
        Initialize Edge TTS Service.
        """
        self.default_voice = voice
        self.voices = {
            'en': 'en-US-ChristopherNeural',
            'hi': 'hi-IN-SwaraNeural',
            'mr': 'mr-IN-AarohiNeural'
        }

    async def generate_speech(self, text, output_file=None, lang='en'):
        """
        Generate speech from text.
        Args:
            text: Text to speak.
            output_file: Path to save audio.
            lang: Language code ('en', 'hi', 'mr')
        """
        if not output_file:
            # Create temp file
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, f"tts_{hash(text)}.mp3")

        voice = self.voices.get(lang, self.default_voice)
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        
        return output_file
