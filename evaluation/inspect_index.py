import chromadb

from rag.config import CHROMA_DIR, COLLECTION_NAME

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(COLLECTION_NAME)

count = collection.count()
print(f"Total chunks in Chroma: {count}")

result = collection.get(
    limit=5,
    include=["documents", "metadatas"],
)

documents = result.get("documents") or []
metadatas = result.get("metadatas") or []

for i, doc in enumerate(documents):
    print("\n" + "=" * 80)

    if i < len(metadatas):
        print(metadatas[i])

    print(str(doc)[:1000])
