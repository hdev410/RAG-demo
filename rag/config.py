import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "source_documents"
INDEX_DIR = ROOT / "artifacts" / "indexes"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

COLLECTION_NAME = "system_function_specs"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVAL_K = 8
FINAL_K = 3
RRF_K = 60
