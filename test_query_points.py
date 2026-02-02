from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np

# Create client
client = QdrantClient(":memory:")

# Create collection
collection_name = "test"
vector_size = 384

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Add test points
test_vectors = [np.random.rand(vector_size).tolist() for _ in range(3)]
points = [
    PointStruct(id=i, vector=vec, payload={"text": f"Document {i}"})
    for i, vec in enumerate(test_vectors)
]

client.upsert(collection_name=collection_name, points=points)

# Test query_points
print("Testing query_points:")
query_vector = np.random.rand(vector_size).tolist()

try:
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=2
    )
    print(f"✅ SUCCESS!")
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print(f"\nResult attributes: {dir(result)}")
    
    if hasattr(result, 'points'):
        print(f"\nPoints: {result.points}")
        for point in result.points:
            print(f"  - ID: {point.id}, Score: {point.score}, Payload: {point.payload}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
