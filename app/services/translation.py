"""
Translation Service for multilingual support
Handles translation between Hindi and English for better LLM processing
"""
from typing import Optional
import asyncio

class TranslationService:
    def __init__(self):
        """
        Initialize translation service.
        Uses a simple approach with language detection and translation.
        """
        self.supported_languages = {
            'hi': 'Hindi',
            'en': 'English',
            'mr': 'Marathi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'ur': 'Urdu'
        }
        
        # Try to import translation libraries
        self.translator = None
        self.translation_available = False
        
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.translation_available = True
            print("✅ Google Translate available")
        except ImportError as e:
            print(f"⚠️  Google Translate not available: {e}")
            print("   Translation will use basic language detection only")
            print("   To enable: pip install googletrans==4.0.0-rc1")
        except Exception as e:
            print(f"⚠️  Google Translate initialization failed: {e}")
            print("   Continuing with basic language detection...")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        Returns language code (e.g., 'hi', 'en')
        """
        if not self.translation_available:
            # Fallback: Simple heuristic based on Unicode ranges
            # Devanagari script (Hindi, Marathi, etc.): U+0900 to U+097F
            devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
            if devanagari_chars > len(text) * 0.3:  # More than 30% Devanagari
                return 'hi'
            return 'en'
        
        try:
            detection = self.translator.detect(text)
            return detection.lang
        except Exception as e:
            print(f"⚠️  Language detection failed: {e}")
            return 'en'
    
    async def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> tuple[str, str]:
        """
        Translate text to English if needed.
        Returns: (translated_text, detected_language)
        """
        if not text or len(text.strip()) == 0:
            return text, 'en'
        
        # Detect language if not provided
        if not source_lang:
            source_lang = self.detect_language(text)
        
        # If already English, return as is
        if source_lang == 'en':
            return text, 'en'
        
        # If translation not available, return original
        if not self.translation_available:
            print(f"⚠️  Translation not available, using original text in {source_lang}")
            return text, source_lang
        
        # Translate to English
        try:
            translation = self.translator.translate(text, src=source_lang, dest='en')
            translated_text = translation.text
            print(f"🌍 Translated from {source_lang} to English: {text[:50]}... → {translated_text[:50]}...")
            return translated_text, source_lang
        except Exception as e:
            print(f"⚠️  Translation failed: {e}")
            return text, source_lang
    
    async def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate English text to target language.
        """
        if not text or len(text.strip()) == 0:
            return text
        
        # If target is English, return as is
        if target_lang == 'en':
            return text
        
        # If translation not available, return original
        if not self.translation_available:
            print(f"⚠️  Translation not available, returning English text")
            return text
        
        # Translate from English to target language
        try:
            translation = self.translator.translate(text, src='en', dest=target_lang)
            translated_text = translation.text
            print(f"🌍 Translated from English to {target_lang}: {text[:50]}... → {translated_text[:50]}...")
            return translated_text
        except Exception as e:
            print(f"⚠️  Translation failed: {e}")
            return text
    
    def get_language_name(self, lang_code: str) -> str:
        """Get the full name of a language from its code."""
        return self.supported_languages.get(lang_code, lang_code.upper())
