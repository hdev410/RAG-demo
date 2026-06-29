from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "System original Function Specs"
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"

COLLECTION_NAME = "system_function_specs"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

INTERFACE_CHUNK_SIZE = 1200
INTERFACE_CHUNK_OVERLAP = 200

STEP_CHUNK_SIZE = 800
STEP_CHUNK_OVERLAP = 120

TOP_K_SEMANTIC = 8
TOP_K_BM25 = 8
FINAL_TOP_K = 3
