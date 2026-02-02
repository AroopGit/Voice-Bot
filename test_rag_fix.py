"""Quick test to verify RAG service is working correctly"""
import sys
sys.path.insert(0, 'd:/Voice_bot')

from app.services.rag import RAGService

print("Testing RAG Service...")
print("=" * 60)

# Initialize RAG service
rag = RAGService(qdrant_path=":memory:")
print("✅ RAG Service initialized successfully")

# Add test documents
test_docs = [
    {"text": "IRCTC stands for Indian Railway Catering and Tourism Corporation."},
    {"text": "You can book train tickets online at www.irctc.co.in."},
    {"text": "PNR status can be checked on the IRCTC website."}
]

rag.add_documents(test_docs)
print("✅ Documents added successfully")

# Test search
query = "How to book tickets?"
results = rag.search(query, limit=2)
print(f"\n✅ Search successful for query: '{query}'")
print(f"Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result['score']:.4f}")
    print(f"   Text: {result['text']}")

print("\n" + "=" * 60)
print("✅ All RAG tests passed!")
