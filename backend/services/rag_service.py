"""
RAG Service - Retrieval Augmented Generation using Qdrant, Sentence-Transformers, and Groq.
Provides high-performance document retrieval and LLM response generation.
"""

import asyncio
import time
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from loguru import logger
import hashlib
from cachetools import TTLCache

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchParams, SearchRequest
)
from sentence_transformers import SentenceTransformer
from groq import Groq, AsyncGroq

try:
    from config import settings, SYSTEM_PROMPT
except ImportError:
    from ..config import settings, SYSTEM_PROMPT


class RAGService:
    """
    High-performance RAG service combining:
    - Qdrant for vector storage and semantic search
    - Sentence-Transformers for embeddings
    - Groq for fast LLM inference
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "swiftship_knowledge",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        groq_api_key: Optional[str] = None
    ):
        """
        Initialize the RAG service.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the vector collection
            embedding_model: Sentence transformer model name
            groq_api_key: Groq API key for LLM
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Clients (initialized lazily)
        self.qdrant_client: Optional[QdrantClient] = None
        self.async_qdrant: Optional[AsyncQdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.groq_client: Optional[Groq] = None
        self.async_groq: Optional[AsyncGroq] = None
        
        # API key
        self.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        
        # Configuration
        self.chunk_size = settings.RAG_CHUNK_SIZE
        self.chunk_overlap = settings.RAG_CHUNK_OVERLAP
        self.top_k = settings.RAG_TOP_K
        self.similarity_threshold = settings.RAG_SIMILARITY_THRESHOLD
        
        # Caching
        self.embedding_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
        self.query_cache = TTLCache(maxsize=500, ttl=300)  # 5 minutes
        
        # Metrics
        self.total_queries = 0
        self.total_retrieval_time = 0.0
        self.total_llm_time = 0.0
        
        self.is_initialized = False
        self._lock = asyncio.Lock()
        
        logger.info(f"RAG Service configured: collection={collection_name}, embedding={embedding_model}")
    
    async def initialize(self) -> bool:
        """Initialize all components."""
        if self.is_initialized:
            return True
            
        async with self._lock:
            if self.is_initialized:
                return True
            
            try:
                start_time = time.perf_counter()
                
                # Load embedding model
                logger.info(f"Loading embedding model: {self.embedding_model_name}...")
                loop = asyncio.get_event_loop()
                self.embedding_model = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(self.embedding_model_name)
                )
                
                # Initialize Qdrant clients - try server first, fallback to in-memory
                try:
                    logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}...")
                    self.qdrant_client = QdrantClient(
                        host=self.qdrant_host,
                        port=self.qdrant_port,
                        timeout=5
                    )
                    # Test connection
                    self.qdrant_client.get_collections()
                    logger.info("Connected to Qdrant server")
                except Exception as e:
                    logger.warning(f"Qdrant server not available: {e}")
                    logger.info("Falling back to in-memory Qdrant (data will not persist)")
                    self.qdrant_client = QdrantClient(":memory:")
                
                # Initialize Groq client
                if self.groq_api_key:
                    self.groq_client = Groq(api_key=self.groq_api_key)
                    self.async_groq = AsyncGroq(api_key=self.groq_api_key)
                    logger.info("Groq client initialized")
                else:
                    logger.warning("No Groq API key provided")
                
                # Ensure collection exists
                await self._ensure_collection()
                
                elapsed = time.perf_counter() - start_time
                logger.info(f"RAG Service initialized in {elapsed:.2f}s")
                self.is_initialized = True
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize RAG service: {e}")
                return False
    
    async def _ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.qdrant_client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                # Get embedding dimension
                test_embedding = self.embedding_model.encode("test")
                vector_size = len(test_embedding)
                
                logger.info(f"Creating collection {self.collection_name} with dim={vector_size}")
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Collection setup failed: {e}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching."""
        # Create cache key from text hash
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = self.embedding_model.encode(text).tolist()
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant document chunks with scores
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        try:
            # Get query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,
                lambda: self._get_embedding(query)
            )
            
            # Search Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold,
                with_payload=True
            )
            
            # Format results
            documents = []
            for result in results:
                documents.append({
                    "id": str(result.id),
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "source": result.payload.get("source", "unknown"),
                    "metadata": result.payload.get("metadata", {})
                })
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self.total_retrieval_time += elapsed
            
            logger.info(f"Retrieved {len(documents)} documents in {elapsed:.0f}ms")
            return documents
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        language: str = "en",
        stream: bool = False,
        use_fast_model: bool = True
    ) -> str | AsyncGenerator[str, None]:
        """
        Generate a response using Groq LLM.
        
        Args:
            query: User query
            context: Retrieved context documents
            language: Response language
            stream: Whether to stream the response
            use_fast_model: Use faster model (8B) vs quality model (70B)
            
        Returns:
            Generated response text or async generator of chunks
        """
        if not self.groq_client:
            return "I apologize, but the AI service is not available right now."
        
        # Build context string
        context_text = self._format_context(context)
        
        # Select model
        model = settings.LLM_MODEL_FAST if use_fast_model else settings.LLM_MODEL_QUALITY
        
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(query, context_text, language)}
        ]
        
        if stream:
            return self._stream_response(messages, model)
        else:
            return await self._generate_sync(messages, model)
    
    async def _generate_sync(self, messages: List[Dict], model: str) -> str:
        """Generate response synchronously."""
        start_time = time.perf_counter()
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.groq_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE
                )
            )
            
            result = response.choices[0].message.content
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self.total_llm_time += elapsed
            self.total_queries += 1
            
            logger.info(f"LLM response generated in {elapsed:.0f}ms ({model})")
            return result
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    async def _stream_response(
        self,
        messages: List[Dict],
        model: str
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response chunks."""
        start_time = time.perf_counter()
        
        try:
            stream = await self.async_groq.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self.total_llm_time += elapsed
            self.total_queries += 1
            logger.info(f"LLM stream completed in {elapsed:.0f}ms")
            
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield "I apologize, but I encountered an error processing your request."
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context documents into a string."""
        if not context:
            return "No relevant information found in the knowledge base."
        
        parts = []
        for i, doc in enumerate(context, 1):
            source = doc.get("source", "Unknown")
            text = doc.get("text", "")
            score = doc.get("score", 0)
            parts.append(f"[Document {i}] (Relevance: {score:.2f}, Source: {source})\n{text}")
        
        return "\n\n".join(parts)
    
    def _build_prompt(self, query: str, context: str, language: str) -> str:
        """Build the complete prompt for the LLM."""
        return f"""Customer Query: {query}

Relevant Information from Knowledge Base:
{context}

Please respond to the customer's query in {language if language != 'en' else 'English'}. 
Be helpful, accurate, and concise. If the information isn't in the knowledge base, 
acknowledge this and offer alternative assistance."""
    
    async def query(
        self,
        query: str,
        language: str = "en",
        stream: bool = False,
        use_fast_model: bool = True
    ) -> Tuple[str, List[Dict]]:
        """
        Complete RAG query: retrieve + generate.
        
        Args:
            query: User query
            language: Response language
            stream: Whether to stream response
            use_fast_model: Use fast or quality model
            
        Returns:
            Tuple of (response, retrieved_documents)
        """
        # Retrieve relevant documents
        documents = await self.retrieve(query)
        
        # Generate response
        response = await self.generate_response(
            query=query,
            context=documents,
            language=language,
            stream=stream,
            use_fast_model=use_fast_model
        )
        
        return response, documents
    
    async def add_document(
        self,
        text: str,
        source: str = "manual",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            text: Document text
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Generate embedding
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._get_embedding(text)
        )
        
        # Generate ID from content hash
        doc_id = hashlib.md5(text.encode()).hexdigest()
        
        # Create point
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload={
                "text": text,
                "source": source,
                "metadata": metadata or {},
                "timestamp": time.time()
            }
        )
        
        # Upsert to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        logger.info(f"Added document {doc_id} from {source}")
        return doc_id
    
    async def add_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple documents in batch.
        
        Args:
            documents: List of dicts with 'text', 'source', 'metadata'
            
        Returns:
            List of document IDs
        """
        if not self.is_initialized:
            await self.initialize()
        
        points = []
        doc_ids = []
        
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
                
            embedding = self._get_embedding(text)
            doc_id = hashlib.md5(text.encode()).hexdigest()
            
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "text": text,
                    "source": doc.get("source", "batch"),
                    "metadata": doc.get("metadata", {}),
                    "timestamp": time.time()
                }
            ))
            doc_ids.append(doc_id)
        
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} documents in batch")
        
        return doc_ids
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        avg_retrieval = self.total_retrieval_time / max(1, self.total_queries)
        avg_llm = self.total_llm_time / max(1, self.total_queries)
        
        return {
            "total_queries": self.total_queries,
            "average_retrieval_ms": avg_retrieval,
            "average_llm_ms": avg_llm,
            "average_total_ms": avg_retrieval + avg_llm,
            "embedding_cache_size": len(self.embedding_cache),
            "collection": self.collection_name,
            "initialized": self.is_initialized
        }
    
    async def health_check(self) -> dict:
        """Check if the service is healthy."""
        status = {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "components": {}
        }
        
        # Check Qdrant
        try:
            if self.qdrant_client:
                info = self.qdrant_client.get_collection(self.collection_name)
                status["components"]["qdrant"] = {
                    "status": "healthy",
                    "points_count": info.points_count
                }
        except Exception as e:
            status["components"]["qdrant"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Groq
        status["components"]["groq"] = {
            "status": "healthy" if self.groq_client else "not_configured"
        }
        
        # Check embedding model
        status["components"]["embedding"] = {
            "status": "healthy" if self.embedding_model else "not_loaded",
            "model": self.embedding_model_name
        }
        
        status["metrics"] = self.get_metrics()
        return status


# Singleton instance
_instance: Optional[RAGService] = None


def get_rag_service(
    qdrant_host: str = None,
    qdrant_port: int = None,
    collection_name: str = None,
    embedding_model: str = None,
    groq_api_key: str = None
) -> RAGService:
    """Get or create the singleton RAG service instance."""
    global _instance
    if _instance is None:
        _instance = RAGService(
            qdrant_host=qdrant_host or settings.QDRANT_HOST,
            qdrant_port=qdrant_port or settings.QDRANT_PORT,
            collection_name=collection_name or settings.QDRANT_COLLECTION,
            embedding_model=embedding_model or settings.EMBEDDING_MODEL,
            groq_api_key=groq_api_key or settings.GROQ_API_KEY
        )
    return _instance
