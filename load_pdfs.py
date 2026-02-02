"""
PDF Loader for RAG System
Loads PDF files from data/ folder and adds them to the vector database
"""

import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from app.services.rag import RAGService

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks for better RAG performance.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size // 2:  # Only break if we're past halfway
                end = start + break_point + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c]  # Remove empty chunks

def load_pdfs_from_folder(folder_path: str = "data") -> List[Dict[str, str]]:
    """
    Load all PDF files from a folder and convert to documents.
    
    Args:
        folder_path: Path to folder containing PDFs
        
    Returns:
        List of document dictionaries with 'text' and metadata
    """
    documents = []
    pdf_folder = Path(folder_path)
    
    if not pdf_folder.exists():
        print(f"⚠️  Folder {folder_path} does not exist. Creating it...")
        pdf_folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created folder: {folder_path}")
        print(f"📁 Place your PDF files in this folder and restart the server.")
        return documents
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {folder_path}")
        return documents
    
    print(f"📚 Found {len(pdf_files)} PDF file(s) in {folder_path}")
    
    for pdf_file in pdf_files:
        print(f"📄 Loading: {pdf_file.name}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(str(pdf_file))
        
        if not text:
            print(f"⚠️  No text extracted from {pdf_file.name}")
            continue
        
        # Chunk the text
        chunks = chunk_text(text)
        print(f"   ✅ Extracted {len(chunks)} chunks from {pdf_file.name}")
        
        # Create documents with metadata
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": pdf_file.name,
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
    
    print(f"✅ Total documents created: {len(documents)}")
    return documents

def load_pdfs_into_rag(rag_service: RAGService, folder_path: str = "data"):
    """
    Load PDFs from folder and add to RAG service.
    
    Args:
        rag_service: RAG service instance
        folder_path: Path to folder containing PDFs
    """
    print("\n" + "="*60)
    print("📚 LOADING PDF DOCUMENTS INTO RAG")
    print("="*60)
    
    documents = load_pdfs_from_folder(folder_path)
    
    if documents:
        print(f"\n📥 Adding {len(documents)} documents to vector database...")
        rag_service.add_documents(documents)
        print("✅ All PDF documents loaded successfully!")
    else:
        print("⚠️  No documents to load. Using sample IRCTC knowledge only.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    # Test the PDF loader
    print("Testing PDF Loader...")
    
    # Create test RAG service
    from app.services.rag import RAGService
    rag = RAGService(qdrant_path=":memory:")
    
    # Load PDFs
    load_pdfs_into_rag(rag, "data")
    
    # Test search
    if rag:
        print("\nTesting search...")
        results = rag.search("railway", limit=3)
        print(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Source: {result['metadata'].get('source', 'unknown')}")
            print(f"   Text: {result['text'][:100]}...")
