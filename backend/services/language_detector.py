"""
Language Detection Service for Hindi, English, and Hinglish.
Optimized for the Indian market with fast, rule-based detection.
"""

import re
import time
from typing import Tuple
from loguru import logger
from cachetools import TTLCache


class LanguageDetectorService:
    """
    Fast, rule-based language detection for Hindi, English, and Hinglish.
    Uses character set analysis and keyword detection for < 100ms performance.
    """
    
    # Supported languages for Indian market
    SUPPORTED_LANGUAGES = {'en', 'hi', 'hi-en', 'mr'}
    
    # Devanagari Unicode range
    DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')
    
    # Latin character pattern (English)
    LATIN_PATTERN = re.compile(r'[a-zA-Z]')
    
    # Strong Hindi indicators (words that are almost exclusively used in Hindi/Hinglish)
    HINDI_INDICATORS = {
        'hai', 'hain', 'ho', 'hoga', 'hogi', 'tha', 'thi',
        'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'pe', 'par',
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
    }
    
    # Marathi indicators (Romanized)
    MARATHI_INDICATORS = {
        'aahe', 'aahet', 'hota', 'hoti', 'hote', 'zalay', 'zhala', 'zali',
        'cha', 'chi', 'che', 'la', 'hun', 'madhe', 'avar', 'varti',
        'kay', 'kevha', 'kuthe', 'kon', 'kase', 'kiti',
        'maza', 'mazi', 'maze', 'tuzha', 'tuzhi', 'tuzhe', 'tumcha', 'tumchi', 'tumche',
        'aapla', 'aapli', 'aaple', 'hyancha', 'tyancha',
        'naahi', 'ho', 'bara', 'thamb', 'jevha', 'tevha', 'karava', 'karavi',
    }
    
    # Context keywords (common in both English and Hinglish logistics)
    # These should NOT trigger Hinglish detection alone
    CONTEXT_KEYWORDS = {
        'delivery', 'order', 'package', 'parcel', 'tracking',
        'cod', 'payment', 'address', 'pin', 'pincode', 'status',
        'number', 'call', 'boy', 'cancel', 'available', 'area',
        'service', 'customer', 'support', 'help', 'tracking'
    }
    
    # Common Hindi greetings and phrases in Latin
    HINDI_GREETINGS_LATIN = {
        'namaste', 'namaskar', 'pranam', 'dhanyawad', 'dhanyavaad',
        'shukriya', 'achha', 'accha', 'theek', 'thik', 'bahut', 'bohot'
    }

    # Marathi greetings (Romanized)
    MARATHI_GREETINGS_LATIN = {
        'namaskar', 'dhanyavad', 'shubh', 'sakhal', 'sandhyakal', 'kasa', 'kay', 'chalalay'
    }
    
    def __init__(self):
        """Initialize the language detector with caching."""
        # Cache for recent detections (TTL of 10 minutes)
        self.cache = TTLCache(maxsize=2000, ttl=600)
        logger.info("Language detector initialized for Hindi/English/Hinglish")
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect if text is English, Hindi, or Hinglish.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence_score)
            - 'en': English
            - 'hi': Hindi (Devanagari script)
            - 'hi-en': Hinglish (code-mixed Hindi-English in Latin script)
        """
        start_time = time.perf_counter()
        
        if not text or len(text.strip()) < 2:
            return "en", 1.0
        
        # Check cache first
        cache_key = text[:200].lower()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            logger.debug(f"Language cache hit: {cached[0]}")
            return cached
        
        # Clean text for analysis
        clean_text = text.strip().lower()
        
        # Count character types
        devanagari_chars = len(self.DEVANAGARI_PATTERN.findall(text))
        latin_chars = len(self.LATIN_PATTERN.findall(text))
        total_alpha = devanagari_chars + latin_chars
        
        if total_alpha == 0:
            result = ("en", 0.5)  # Default to English for non-text
        else:
            devanagari_ratio = devanagari_chars / total_alpha
            latin_ratio = latin_chars / total_alpha
            
            # Rule-based detection
            if devanagari_ratio > 0.5:
                # Mostly Devanagari - check for Marathi specific markers in raw text if possible
                # For now, if we see 'आहे' or 'का' at end, it's likely Marathi
                if re.search(r'[आहे|ला|चा|ची|चे]\s*$', text):
                    result = ("mr", 0.95)
                else:
                    result = ("hi", 0.95)
            elif devanagari_ratio > 0.1:
                # Some Devanagari = Hinglish
                result = ("hi-en", 0.9)
            else:
                # Pure Latin script - check for Romanized Hindi/Marathi
                hinglish_score = self._check_hinglish_keywords(clean_text)
                marathi_score = self._check_marathi_keywords(clean_text)
                
                if marathi_score > hinglish_score and marathi_score > 0.2:
                    result = ("mr", 0.8 + marathi_score * 0.15)
                elif hinglish_score > 0.3:
                    result = ("hi-en", 0.8 + hinglish_score * 0.15)
                elif hinglish_score > 0.15:
                    result = ("hi-en", 0.7)
                elif marathi_score > 0.15:
                    result = ("mr", 0.7)
                else:
                    result = ("en", 0.9)
        
        # Cache the result
        self.cache[cache_key] = result
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Language: {result[0]} (conf={result[1]:.2f}) in {elapsed:.1f}ms | '{text[:50]}...'")
        
        return result
    
    def _check_hinglish_keywords(self, text: str) -> float:
        """Check for Romanized Hindi keywords in Latin text."""
        words = set(re.findall(r'\b[a-z]+\b', text))
        if not words: return 0.0
        
        hindi_matches = words & self.HINDI_INDICATORS
        greeting_matches = words & self.HINDI_GREETINGS_LATIN
        context_matches = words & self.CONTEXT_KEYWORDS
        
        core_matches = len(hindi_matches) + len(greeting_matches) * 2
        
        if core_matches > 0:
            total_matches = core_matches + len(context_matches) * 0.5
        else:
            total_matches = 0.0
            
        return min(1.0, total_matches / max(2, len(words) * 0.4))

    def _check_marathi_keywords(self, text: str) -> float:
        """Check for Romanized Marathi keywords in Latin text."""
        words = set(re.findall(r'\b[a-z]+\b', text))
        if not words: return 0.0
        
        mr_matches = words & self.MARATHI_INDICATORS
        greet_matches = words & self.MARATHI_GREETINGS_LATIN
        context_matches = words & self.CONTEXT_KEYWORDS
        
        core_matches = len(mr_matches) + len(greet_matches) * 2
        
        if core_matches > 0:
            total_matches = core_matches + len(context_matches) * 0.5
        else:
            total_matches = 0.0
            
        return min(1.0, total_matches / max(2, len(words) * 0.4))
    
    def get_supported_languages(self) -> set:
        """Return set of supported language codes."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def is_supported(self, lang_code: str) -> bool:
        """Check if a language code is supported."""
        return lang_code in self.SUPPORTED_LANGUAGES
    
    def get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name."""
        names = {
            'en': 'English',
            'hi': 'Hindi (हिंदी)',
            'hi-en': 'Hinglish',
            'mr': 'Marathi (मराठी)'
        }
        return names.get(lang_code, 'English')
    
    async def detect_async(self, text: str) -> Tuple[str, float]:
        """Async wrapper for language detection."""
        return self.detect(text)
    
    def health_check(self) -> dict:
        """Check if the service is healthy with test cases."""
        test_cases = {
            "en": "Where is my package? What is the delivery status?",
            "hi": "मेरा पैकेज कहाँ है? डिलीवरी कब होगी?",
            "hi-en": "Mera package kaha hai? Delivery kab hogi?",
            "mr": "Majha package kuthe aahe? Delivery kadhi hoil?",
        }
        
        results = {}
        for expected, text in test_cases.items():
            detected, confidence = self.detect(text)
            results[expected] = {
                "text": text[:30] + "...",
                "detected": detected,
                "matched": detected == expected,
                "confidence": round(confidence, 2)
            }
        
        all_matched = all(r["matched"] for r in results.values())
        
        return {
            "status": "healthy" if all_matched else "degraded",
            "supported_languages": list(self.SUPPORTED_LANGUAGES),
            "test_results": results,
            "cache_size": len(self.cache)
        }


# Singleton instance
_instance = None


def get_language_detector() -> LanguageDetectorService:
    """Get or create the singleton language detector instance."""
    global _instance
    if _instance is None:
        _instance = LanguageDetectorService()
    return _instance
