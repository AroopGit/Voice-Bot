from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class RAGService:
    def __init__(self, collection_name="knowledge_base", embedding_model="all-MiniLM-L6-v2", qdrant_path=":memory:"):
        """
        Initialize RAG Service.
        Args:
            collection_name: Name of the Qdrant collection.
            embedding_model: Name of the sentence-transformer model.
            qdrant_path: Path to Qdrant storage (or ":memory:", "http://localhost:6333").
        """
        print(f"Loading embedding model: {embedding_model}...")
        self.encoder = SentenceTransformer(embedding_model)
        print("Embedding model loaded.")

        self.client = QdrantClient(path=qdrant_path) if qdrant_path != ":memory:" else QdrantClient(":memory:")
        self.collection_name = collection_name
        
        # Create collection with proper VectorParams model
        vector_size = self.encoder.get_sentence_embedding_dimension()
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the knowledge base.
        documents: List of dicts with 'text' field and metadata.
        """
        if not documents:
            return

        texts = [doc['text'] for doc in documents]
        embeddings = self.encoder.encode(texts).tolist()
        
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            payload = documents[i]
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload=payload
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )


    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search for relevant documents using vector similarity.
        """
        query_vector = self.encoder.encode(query).tolist()
        
        # Use query_points (correct method for Qdrant 1.16+)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        
        results = []
        for hit in result.points:
            results.append({
                "text": hit.payload.get('text', ''),
                "score": hit.score,
                "metadata": hit.payload
            })
        return results
