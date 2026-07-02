import pickle
import shutil

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.config import BM25_INDEX_PATH, CHROMA_DIR, COLLECTION_NAME, DATA_DIR, EMBEDDING_MODEL, INDEX_DIR
from rag.ingestion.chunking import chunk_pages
from rag.ingestion.loader import load_pdfs
from rag.shared.text import tokenize


def main() -> None:
    """Run the offline RAG pipeline and persist both search indexes."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Document folder not found: {DATA_DIR}")

    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    BM25_INDEX_PATH.unlink(missing_ok=True)

    # 1. Load PDFs and create small, overlapping retrieval units.
    chunks = chunk_pages(load_pdfs(DATA_DIR))
    if not chunks:
        raise RuntimeError("No chunks were created")

    texts = [chunk["text"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]
    ids = [str(index) for index in range(len(chunks))]
    # 2. Convert chunk meaning into dense embedding vectors.
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Store vectors and metadata in ChromaDB for semantic search.
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).create_collection(COLLECTION_NAME)

    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        collection.add(
            ids=ids[start : start + 64],
            documents=batch,
            metadatas=metadata[start : start + 64],
            embeddings=model.encode(batch, show_progress_bar=True).tolist(),
        )

    # 4. Build a lexical BM25 index for exact words, IDs, and names.
    with BM25_INDEX_PATH.open("wb") as file:
        pickle.dump({"bm25": BM25Okapi([tokenize(text) for text in texts]), "texts": texts, "metadata": metadata, "ids": ids}, file)
    print(f"Indexed {len(chunks)} chunks in {INDEX_DIR}")


if __name__ == "__main__":
    main()
