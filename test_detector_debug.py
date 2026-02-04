import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "backend"))

from services.language_detector import get_language_detector

detector = get_language_detector()
test_cases = [
    ("Where is my package?", "en"),
    ("मेरा पैकेज कहाँ है?", "hi"),
    ("Mera package kaha hai?", "hi-en"),
]

for text, expected in test_cases:
    detected, conf = detector.detect(text)
    print(f"Text: '{text}' | Expected: {expected} | Detected: {detected} | Conf: {conf:.2f}")
    if detected != expected:
        print(f"FAILED: Expected {expected}, got {detected}")
