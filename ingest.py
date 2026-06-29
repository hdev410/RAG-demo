import pickle
import shutil
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    CHROMA_DIR,
    BM25_INDEX_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from pdf_loader import load_all_pdfs
from chunk_interface import chunk_interface_pages
from chunk_step import chunk_step_pages
from text_utils import tokenize


def group_pages_by_document(pages: list[dict]) -> dict:
    grouped = defaultdict(list)

    for page in pages:
        key = (page["group"], page["source"])
        grouped[key].append(page)

    return grouped


def main():
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    if BM25_INDEX_PATH.exists():
        BM25_INDEX_PATH.unlink()

    pages = load_all_pdfs(DATA_DIR)
    grouped_docs = group_pages_by_document(pages)

    all_chunks = []

    for (group, source), doc_pages in grouped_docs.items():
        print(f"Chunking [{group}]: {source}")

        if group == "interface":
            chunks = chunk_interface_pages(doc_pages)
        elif group == "step":
            chunks = chunk_step_pages(doc_pages)
        else:
            continue

        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    ids = []
    texts = []
    metadatas = []

    for index, chunk in enumerate(all_chunks):
        ids.append(str(index))
        texts.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    batch_size = 64

    for start in range(0, len(texts), batch_size):
        end = start + batch_size

        batch_texts = texts[start:end]
        batch_embeddings = embedding_model.encode(
            batch_texts,
            show_progress_bar=True
        ).tolist()

        collection.add(
            ids=ids[start:end],
            documents=batch_texts,
            metadatas=metadatas[start:end],
            embeddings=batch_embeddings
        )

    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as file:
        pickle.dump({
            "bm25": bm25,
            "texts": texts,
            "metadatas": metadatas,
            "ids": ids
        }, file)

    print("Ingestion completed.")
    print(f"Chroma DB saved to: {CHROMA_DIR}")
    print(f"BM25 index saved to: {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()