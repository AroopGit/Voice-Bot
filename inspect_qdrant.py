from qdrant_client import QdrantClient
import inspect

client = QdrantClient(":memory:")

# Get all public methods
methods = [m for m in dir(client) if not m.startswith('_') and callable(getattr(client, m))]

print("All public methods:")
print("=" * 60)
for method in sorted(methods):
    print(f"  - {method}")

print("\n" + "=" * 60)
print("Methods containing 'search':")
search_methods = [m for m in methods if 'search' in m.lower()]
for method in search_methods:
    print(f"  - {method}")
    try:
        sig = inspect.signature(getattr(client, method))
        print(f"    Signature: {sig}")
    except:
        pass

print("\n" + "=" * 60)
print("Methods containing 'query':")
query_methods = [m for m in methods if 'query' in m.lower()]
for method in query_methods:
    print(f"  - {method}")
    try:
        sig = inspect.signature(getattr(client, method))
        print(f"    Signature: {sig}")
    except:
        pass
