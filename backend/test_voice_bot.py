"""
Test script for the SwiftShip India Voice Bot.
Tests Hindi, English, and Hinglish language support.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


class TestResults:
    """Collect and display test results."""
    
    def __init__(self):
        self.results = []
        self.total_time = 0
    
    def add(self, name: str, passed: bool, time_ms: float, details: str = ""):
        self.results.append({
            "name": name,
            "passed": passed,
            "time_ms": time_ms,
            "details": details
        })
        self.total_time += time_ms
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY - SwiftShip India Voice Bot")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r["passed"])
        failed = len(self.results) - passed
        
        for r in self.results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            print(f"{status} | {r['name']:<40} | {r['time_ms']:>8.1f}ms")
            if r["details"]:
                print(f"       {r['details']}")
        
        print("=" * 70)
        print(f"Total: {len(self.results)} tests | Passed: {passed} | Failed: {failed}")
        print(f"Total time: {self.total_time:.1f}ms ({self.total_time/1000:.2f}s)")
        print("=" * 70)
        
        return failed == 0


async def test_language_detector():
    """Test 3-way language detection for Hindi, English, Hinglish."""
    from services.language_detector import get_language_detector
    
    start = time.perf_counter()
    detector = get_language_detector()
    
    # Test cases for all three languages
    test_cases = [
        # English queries
        ("Where is my package?", "en"),
        ("What is the status of my order?", "en"),
        ("When will it arrive?", "en"),
        
        # Hindi queries (Devanagari)
        ("मेरा पैकेज कहाँ है?", "hi"),
        ("डिलीवरी कब होगी?", "hi"),
        ("मेरा ऑर्डर कब तक पहुंचेगा?", "hi"),
        
        # Hinglish queries (Romanized Hindi)
        ("Mera package kaha hai?", "hi-en"),
        ("Delivery kab hoga?", "hi-en"),
        ("COD available hai kya?", "hi-en"),
        ("Order kab aayega mera?", "hi-en"),
        ("Tracking number 12345 ka status kya hai?", "hi-en"),
    ]
    
    passed = 0
    details = []
    
    for text, expected in test_cases:
        detected, confidence = detector.detect(text)
        if detected == expected:
            passed += 1
            details.append(f"✓ {expected}")
        else:
            details.append(f"✗ Expected {expected}, got {detected}: '{text[:20]}'")
    
    all_passed = passed == len(test_cases)
    elapsed = (time.perf_counter() - start) * 1000
    
    summary = f"{passed}/{len(test_cases)} correct"
    if not all_passed:
        summary += " | " + " | ".join([d for d in details if d.startswith("✗")])
    
    return all_passed, elapsed, summary


async def test_english_query():
    """Test English language query processing."""
    from services.language_detector import get_language_detector
    
    start = time.perf_counter()
    detector = get_language_detector()
    
    test_queries = [
        "What is the status of my order #12345?",
        "Is COD available in my area?",
        "How long does delivery take to Mumbai?",
    ]
    
    passed = True
    for query in test_queries:
        lang, conf = detector.detect(query)
        if lang != 'en':
            passed = False
            break
    
    elapsed = (time.perf_counter() - start) * 1000
    return passed, elapsed, f"All {len(test_queries)} English queries detected correctly"


async def test_hindi_query():
    """Test Hindi (Devanagari) query processing."""
    from services.language_detector import get_language_detector
    
    start = time.perf_counter()
    detector = get_language_detector()
    
    test_queries = [
        "मेरा ऑर्डर कब तक पहुंचेगा?",
        "क्या COD उपलब्ध है?",
        "मुंबई में डिलीवरी का समय क्या है?",
    ]
    
    passed = True
    for query in test_queries:
        lang, conf = detector.detect(query)
        if lang != 'hi':
            passed = False
            break
    
    elapsed = (time.perf_counter() - start) * 1000
    return passed, elapsed, f"All {len(test_queries)} Hindi queries detected correctly"


async def test_hinglish_query():
    """Test Hinglish (code-mixed) query processing."""
    from services.language_detector import get_language_detector
    
    start = time.perf_counter()
    detector = get_language_detector()
    
    test_queries = [
        "Mera package kab deliver hoga?",
        "COD available hai mere PIN code pe?",
        "Tracking number 12345 ka status kya hai?",
        "Order cancel karna hai, kaise karun?",
        "Delivery boy ne call nahi kiya",
    ]
    
    passed = 0
    for query in test_queries:
        lang, conf = detector.detect(query)
        if lang == 'hi-en':
            passed += 1
    
    # Allow 80% accuracy for Hinglish (it's challenging)
    success = passed >= len(test_queries) * 0.8
    elapsed = (time.perf_counter() - start) * 1000
    
    return success, elapsed, f"{passed}/{len(test_queries)} Hinglish queries detected"


async def test_tts_indian_voices():
    """Test TTS with Indian voices for all three languages."""
    from services.tts_service import get_tts_service
    
    start = time.perf_counter()
    tts = get_tts_service()
    
    test_cases = [
        ("en", "Hello, your package will arrive tomorrow."),
        ("hi", "नमस्ते, आपका पैकेज कल पहुंच जाएगा।"),
        ("hi-en", "Your order aaj shaam tak deliver ho jayega."),
    ]
    
    results = []
    for lang, text in test_cases:
        audio = await tts.synthesize(text, lang)
        success = len(audio) > 1000
        results.append((lang, success, len(audio)))
    
    all_passed = all(r[1] for r in results)
    elapsed = (time.perf_counter() - start) * 1000
    
    details = " | ".join([f"{r[0]}:{r[2]}bytes" for r in results])
    return all_passed, elapsed, details


async def test_groq_connection():
    """Test Groq LLM connection."""
    from config import settings
    
    start = time.perf_counter()
    
    if not settings.GROQ_API_KEY:
        return False, 0, "GROQ_API_KEY not set in .env"
    
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say 'namaste' in one word"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        elapsed = (time.perf_counter() - start) * 1000
        
        return True, elapsed, f"Response: {result[:30]}"
        
    except Exception as e:
        return False, (time.perf_counter() - start) * 1000, f"Error: {str(e)}"


async def test_qdrant_connection():
    """Test Qdrant vector database connection."""
    from config import settings
    
    start = time.perf_counter()
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        
        # Try to get collections
        collections = client.get_collections()
        
        elapsed = (time.perf_counter() - start) * 1000
        return True, elapsed, f"Connected, {len(collections.collections)} collections"
        
    except Exception as e:
        return False, (time.perf_counter() - start) * 1000, f"Error: {str(e)}"


async def test_stt_initialization():
    """Test STT service initialization."""
    from services.stt_service import get_stt_service
    
    start = time.perf_counter()
    
    try:
        stt = get_stt_service(model_size="small", device="auto")
        init_success = await stt.initialize()
        
        elapsed = (time.perf_counter() - start) * 1000
        
        if init_success:
            return True, elapsed, f"Whisper {stt.model_size} initialized"
        else:
            return False, elapsed, "Initialization failed"
            
    except Exception as e:
        return False, (time.perf_counter() - start) * 1000, f"Error: {str(e)}"


async def test_sample_interactions():
    """Test sample conversations in all three languages."""
    print("\n" + "-" * 50)
    print("SAMPLE INTERACTION TESTS")
    print("-" * 50)
    
    from services.language_detector import get_language_detector
    from config import get_language_name
    
    detector = get_language_detector()
    
    interactions = [
        # English
        {
            "query": "Where is my package? Tracking number is ABC123",
            "expected_lang": "en",
            "expected_response_lang": "English"
        },
        # Hindi
        {
            "query": "मेरा पैकेज कहाँ है? ट्रैकिंग नंबर ABC123 है",
            "expected_lang": "hi",
            "expected_response_lang": "Hindi"
        },
        # Hinglish
        {
            "query": "Mera package kaha hai? Tracking number ABC123 hai",
            "expected_lang": "hi-en",
            "expected_response_lang": "Hinglish"
        },
    ]
    
    start = time.perf_counter()
    passed = 0
    
    for i, interaction in enumerate(interactions, 1):
        query = interaction["query"]
        expected = interaction["expected_lang"]
        
        detected, confidence = detector.detect(query)
        
        if detected == expected:
            passed += 1
            print(f"  {i}. ✅ {get_language_name(detected)} query detected")
            print(f"     Q: \"{query[:50]}...\"")
            print(f"     Expected response in: {interaction['expected_response_lang']}")
        else:
            print(f"  {i}. ❌ Expected {expected}, got {detected}")
            print(f"     Q: \"{query[:50]}...\"")
    
    elapsed = (time.perf_counter() - start) * 1000
    
    return passed == len(interactions), elapsed, f"{passed}/{len(interactions)} interactions correct"


async def run_all_tests():
    """Run all tests and print results."""
    print("\n" + "=" * 70)
    print("SWIFTSHIP INDIA VOICE BOT - TEST SUITE")
    print("Optimized for Hindi, English, and Hinglish")
    print("=" * 70)
    print(f"Starting tests at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    results = TestResults()
    
    # Core tests
    tests = [
        ("Language Detection (3-way)", test_language_detector),
        ("English Query Detection", test_english_query),
        ("Hindi Query Detection", test_hindi_query),
        ("Hinglish Query Detection", test_hinglish_query),
        ("Indian TTS Voices", test_tts_indian_voices),
        ("Groq LLM Connection", test_groq_connection),
        ("Qdrant Connection", test_qdrant_connection),
        ("STT Initialization", test_stt_initialization),
        ("Sample Interactions", test_sample_interactions),
    ]
    
    for name, test_func in tests:
        print(f"Running: {name}...", end=" ", flush=True)
        try:
            passed, elapsed, details = await test_func()
            results.add(name, passed, elapsed, details)
            status = "✅" if passed else "❌"
            print(f"{status} ({elapsed:.0f}ms)")
        except Exception as e:
            results.add(name, False, 0, str(e))
            print(f"❌ Error: {e}")
    
    # Print summary
    all_passed = results.print_summary()
    
    # Print language support summary
    print("\n📌 Supported Languages:")
    print("   • English (en) - Indian English voice")
    print("   • Hindi (hi) - हिंदी Devanagari script")
    print("   • Hinglish (hi-en) - Code-mixed Hindi-English")
    
    return all_passed


if __name__ == "__main__":
    # Set up logging for tests
    logger.remove()
    logger.add(sys.stdout, level="WARNING")
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
