"""
Translation Service for multilingual support
Enhanced for Hindi, English, and Marathi accuracy
Uses deep_translator for reliable translations
"""
from typing import Optional, Tuple
import asyncio
import re

class TranslationService:
    def __init__(self):
        """
        Initialize translation service with enhanced language detection.
        Uses deep_translator for reliable translations and langdetect for detection.
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
        
        # Hindi-specific words (Romanized and common patterns)
        self.hindi_words = {
            'hai', 'hain', 'ho', 'hoga', 'hogi', 'tha', 'thi', 'the',
            'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'pe', 'par', 'tak',
            'kya', 'kab', 'kaha', 'kahan', 'kaun', 'kaise', 'kitna', 'kitne', 'kitni',
            'mera', 'meri', 'mere', 'tera', 'teri', 'tere', 'aapka', 'aapki', 'aapke',
            'yeh', 'ye', 'woh', 'wo', 'iska', 'iski', 'uska', 'uski',
            'aur', 'ya', 'lekin', 'toh', 'bhi', 'sirf', 'bas',
            'abhi', 'aaj', 'kal', 'parso', 'baad', 'pehle', 'jab', 'tab',
            'nahi', 'nahin', 'nhi', 'mat', 'bilkul', 'haan', 'ji',
            'karo', 'kardo', 'batao', 'batado', 'bhejo', 'bhejdo',
            'milega', 'milegi', 'aayega', 'aayegi', 'pahunchega', 'pahunchegi',
            'chahiye', 'chahte', 'chahti', 'lena', 'dena', 'rakhna',
            'karoge', 'karenge', 'doge', 'denge', 'hojayega',
            'mujhe', 'humein', 'tumhein', 'aapko', 'unko', 'isko', 'usko',
            'karna', 'karana', 'banana', 'dekhna', 'sunna',
            'namaste', 'namaskar', 'dhanyawad', 'dhanyavaad', 'shukriya',
            'achha', 'accha', 'theek', 'thik', 'bahut', 'bohot',
            'bataiye', 'bataye', 'dijiye', 'kijiye', 'lijiye'
        }
        
        # Marathi-specific words (Romanized)
        self.marathi_words = {
            'aahe', 'aahet', 'ahe', 'hota', 'hoti', 'hote', 'zalay', 'zhala', 'zali', 'zale',
            'cha', 'chi', 'che', 'la', 'hun', 'madhe', 'madhye', 'avar', 'varti', 'khali',
            'kay', 'kevha', 'kuthe', 'kon', 'kasa', 'kashi', 'kiti', 'kashi',
            'maza', 'mazi', 'maze', 'tuzha', 'tuzhi', 'tuzhe', 'tumcha', 'tumchi', 'tumche',
            'aapla', 'aapli', 'aaple', 'hyancha', 'tyancha', 'tyachi', 'hichi',
            'nahi', 'naahi', 'ho', 'hoy', 'bara', 'barr', 'thamb', 'jevha', 'tevha',
            'karava', 'karavi', 'karaycha', 'karaychi', 'karaychay',
            'pahije', 'pahijet', 'hava', 'havi', 'have', 'lage', 'lagte', 'lagel',
            'pathavaycha', 'pathav', 'pathva', 'dya', 'ghya', 'ya', 'ja',
            'sanga', 'sangta', 'sangitla', 'aikla', 'aikla', 'bagha', 'pahila',
            'mhanun', 'mhantat', 'aamhi', 'tumhi', 'te', 'ti', 'tyanna',
            'paryant', 'sathi', 'karun', 'hoil', 'honar', 'asel', 'naste',
            'namaskar', 'dhanyavad', 'shubh'
        }
        
        # Common logistics terms that appear in all languages
        self.logistics_terms = {
            'truck', 'container', 'tempo', 'trailer', 'lorry',
            'booking', 'book', 'rate', 'price', 'cost', 'quote',
            'track', 'tracking', 'shipment', 'order', 'delivery',
            'mumbai', 'delhi', 'pune', 'bangalore', 'chennai', 'hyderabad',
            'kolkata', 'jaipur', 'ahmedabad', 'surat', 'lucknow'
        }
        
        # Try to import translation libraries
        self.translator = None
        self.lang_detector = None
        self.translation_available = False
        
        # Try deep_translator (more reliable)
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator
            self.translation_available = True
            print("✅ Deep Translator (Google) available for translations")
        except ImportError as e:
            print(f"⚠️  Deep Translator not available: {e}")
        
        # Try langdetect for better language detection
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0  # Consistent results
            self.lang_detector = detect
            print("✅ LangDetect available for language detection")
        except ImportError as e:
            print(f"⚠️  LangDetect not available: {e}")
            print("   Using script-based detection only")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text with HIGH ACCURACY.
        Uses script detection + word matching + langdetect for best results.
        Returns language code (e.g., 'hi', 'en', 'mr')
        """
        if not text or len(text.strip()) < 2:
            return 'en'
        
        text_lower = text.lower()
        
        # Step 1: Check for Devanagari script (Hindi/Marathi)
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_chars = len(text.replace(' ', ''))
        devanagari_ratio = devanagari_chars / max(1, total_chars)
        
        # If significant Devanagari content, determine Hindi vs Marathi
        if devanagari_ratio > 0.3:
            # Check for Marathi-specific markers
            marathi_markers = ['आहे', 'आहेत', 'ला', 'मध्ये', 'साठी', 'पाठव', 'हवा', 'हवी', 'हवे',
                              'करायचा', 'करायची', 'पर्यंत', 'कुठे', 'कधी', 'होईल', 'होणार']
            hindi_markers = ['है', 'हैं', 'था', 'थी', 'को', 'में', 'से', 
                            'के', 'की', 'का', 'चाहिए', 'करना', 'होगा', 'होगी',
                            'कहाँ', 'कब', 'क्या', 'कैसे', 'कितना']
            
            marathi_score = sum(1 for m in marathi_markers if m in text)
            hindi_score = sum(1 for h in hindi_markers if h in text)
            
            # Specific Marathi endings
            if text.strip().endswith('आहे') or text.strip().endswith('आहेत'):
                marathi_score += 3
            if 'पर्यंत' in text or 'साठी' in text:
                marathi_score += 2
            
            # Specific Hindi patterns
            if 'चाहिए' in text or 'करना है' in text:
                hindi_score += 3
            if 'में' in text and 'मध्ये' not in text:
                hindi_score += 2
            
            if marathi_score > hindi_score:
                return 'mr'
            else:
                return 'hi'
        
        # Step 2: Check for Romanized Hindi/Marathi words
        words = set(re.findall(r'\b[a-zA-Z]+\b', text_lower))
        
        # Remove common logistics terms for language detection
        words_for_detection = words - self.logistics_terms
        
        hindi_matches = len(words_for_detection & self.hindi_words)
        marathi_matches = len(words_for_detection & self.marathi_words)
        
        # Calculate match ratios
        total_words = len(words_for_detection)
        if total_words > 0:
            hindi_ratio = hindi_matches / total_words
            marathi_ratio = marathi_matches / total_words
        else:
            hindi_ratio = 0
            marathi_ratio = 0
        
        # Decision based on word matches
        if marathi_ratio > 0.15 and marathi_matches >= 2:
            return 'mr'
        if hindi_ratio > 0.15 and hindi_matches >= 2:
            return 'hi'
        if marathi_matches > hindi_matches and marathi_matches >= 2:
            return 'mr'
        if hindi_matches > marathi_matches and hindi_matches >= 2:
            return 'hi'
        
        # Step 3: Use langdetect if available
        if self.lang_detector:
            try:
                detected = self.lang_detector(text)
                # Map to our supported languages
                if detected in ['hi', 'mr', 'en']:
                    return detected
                # If detected as close language, map appropriately
                if detected in ['gu', 'pa', 'bn', 'ur', 'ne']:
                    return 'hi'  # Treat as Hindi for processing
            except Exception:
                pass
        
        return 'en'
    
    async def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> Tuple[str, str]:
        """
        Translate text to English if needed.
        Returns: (translated_text, detected_language)
        """
        if not text or len(text.strip()) == 0:
            return text, 'en'
        
        # Detect language if not provided
        if not source_lang:
            source_lang = self.detect_language(text)
        
        print(f"🔍 Language detection result: {source_lang} ({self.get_language_name(source_lang)})")
        
        # If already English, return as is
        if source_lang == 'en':
            return text, 'en'
        
        # If translation not available, return original with detected language
        if not self.translation_available:
            print(f"⚠️  Translation not available, using original {source_lang} text for processing")
            return text, source_lang
        
        # Translate to English using deep_translator
        try:
            translator = self.translator(source=source_lang, target='en')
            translated_text = translator.translate(text)
            if translated_text:
                print(f"🌍 Translated from {source_lang}: '{text[:60]}...' → '{translated_text[:60]}...'")
                return translated_text, source_lang
            return text, source_lang
        except Exception as e:
            print(f"⚠️  Translation failed: {e}")
            return text, source_lang
    
    async def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate English text to target language (Hindi/Marathi).
        """
        if not text or len(text.strip()) == 0:
            return text
        
        # If target is English, return as is
        if target_lang == 'en':
            return text
        
        # If translation not available, return English
        if not self.translation_available:
            print(f"⚠️  Translation to {target_lang} not available, returning English")
            return text
        
        # Translate from English to target language
        try:
            translator = self.translator(source='en', target=target_lang)
            translated_text = translator.translate(text)
            if translated_text:
                print(f"🌍 Translated to {target_lang}: '{text[:50]}...' → '{translated_text[:50]}...'")
                return translated_text
            return text
        except Exception as e:
            print(f"⚠️  Translation to {target_lang} failed: {e}")
            return text
    
    def get_language_name(self, lang_code: str) -> str:
        """Get the full name of a language from its code."""
        return self.supported_languages.get(lang_code, lang_code.upper())
    
    def is_devanagari(self, text: str) -> bool:
        """Check if text contains significant Devanagari script."""
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        return devanagari_chars > len(text.replace(' ', '')) * 0.3
