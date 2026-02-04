import requests
import json

base_url = "http://localhost:8000/api"

queries = [
    {"text": "Where is my package?", "language": "en"},
    {"text": "मेरा पैकेज कहाँ है?", "language": "hi"},
    {"text": "Mera package kab deliver hoga?", "language": "hi-en"}
]

for query in queries:
    print(f"\nSending Query: {query['text']} ({query['language']})")
    try:
        response = requests.post(
            f"{base_url}/text-query",
            json=query,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"Response ({result['language']}): {result['response']}")
            print(f"Docs found: {result['documents_found']}")
            print(f"Processing time: {result['processing_time_ms']}ms")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
