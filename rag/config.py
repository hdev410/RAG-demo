import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "source_documents"
INDEX_DIR = ROOT / "artifacts" / "indexes"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

COLLECTION_NAME = "system_function_specs"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking parameters: rebuild the index after changing these values.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval parameters: tune these without changing retrieval code.
TOP_K_SEMANTIC = 8
TOP_K_BM25 = 8
FINAL_TOP_K = 3
RRF_K = 60

# Optional second-stage reranking. It is disabled for the basic demo.
RERANK_ENABLED = False
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
