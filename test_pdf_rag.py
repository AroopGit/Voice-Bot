"""
Quick test script to verify PDF RAG system is working
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.rag import RAGService
from load_pdfs import load_pdfs_into_rag

def test_pdf_rag():
    """Test the PDF RAG system"""
    
    print("\n" + "="*70)
    print("🧪 TESTING PDF RAG SYSTEM")
    print("="*70 + "\n")
    
    # Step 1: Initialize RAG service
    print("1️⃣  Initializing RAG service...")
    rag = RAGService(qdrant_path=":memory:")
    print("   ✅ RAG service initialized\n")
    
    # Step 2: Load PDFs
    print("2️⃣  Loading PDFs from data/ folder...")
    load_pdfs_into_rag(rag, "data")
    print()
    
    # Step 3: Test searches
    print("3️⃣  Testing search functionality...\n")
    
    test_queries = [
        "refund rules",
        "cancellation policy",
        "train ticket",
        "IRCTC booking"
    ]
    
    for query in test_queries:
        print(f"   🔍 Query: '{query}'")
        results = rag.search(query, limit=3)
        
        if results:
            print(f"   ✅ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                source = result['metadata'].get('source', 'unknown')
                score = result['score']
                text_preview = result['text'][:80].replace('\n', ' ')
                print(f"      {i}. [{source}] Score: {score:.3f}")
                print(f"         \"{text_preview}...\"")
        else:
            print(f"   ⚠️  No results found")
        print()
    
    # Step 4: Summary
    print("="*70)
    print("✅ PDF RAG SYSTEM TEST COMPLETE")
    print("="*70)
    print("\nIf you see PDF content above, the system is working!")
    print("If not, check:")
    print("  1. PDF file exists in data/ folder")
    print("  2. PDF contains readable text (not scanned images)")
    print("  3. PyPDF2 is installed: pip install PyPDF2")
    print()

if __name__ == "__main__":
    try:
        test_pdf_rag()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
