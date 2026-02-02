from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np

# Create a test client
client = QdrantClient(":memory:")

# Create a test collection
collection_name = "test"
vector_size = 384

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Add a test point
test_vector = np.random.rand(vector_size).tolist()
client.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(
            id=1,
            vector=test_vector,
            payload={"text": "test document"}
        )
    ]
)

# Try different search methods
print("Testing search methods:")
print("\n1. Trying 'search' method:")
try:
    result = client.search(
        collection_name=collection_name,
        query_vector=test_vector,
        limit=1
    )
    print(f"   ✅ SUCCESS: {type(result)}")
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n2. Trying 'query_points' method:")
try:
    result = client.query_points(
        collection_name=collection_name,
        query=test_vector,
        limit=1
    )
    print(f"   ✅ SUCCESS: {type(result)}")
    print(f"   Result structure: {dir(result)}")
    if hasattr(result, 'points'):
        print(f"   Points: {result.points}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n3. Checking available methods:")
search_methods = [m for m in dir(client) if 'search' in m.lower() and not m.startswith('_')]
print(f"   Available search methods: {search_methods}")
