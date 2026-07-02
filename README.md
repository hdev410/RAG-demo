# Beginner RAG Demo

This project is a small Retrieval-Augmented Generation (RAG) pipeline for PDF documents. It intentionally uses plain Python modules so each step is easy to inspect and explain.

## How the pipeline works

### Offline indexing

1. `loader.py` extracts text and page metadata from each PDF.
2. `chunking.py` groups text by section heading and creates overlapping chunks.
3. A Sentence Transformer converts each chunk into an embedding.
4. ChromaDB stores embeddings for semantic search.
5. BM25 stores token statistics for keyword search.

### Online query

1. Semantic search finds chunks with similar meaning.
2. BM25 finds chunks containing strong exact keyword matches.
3. Reciprocal Rank Fusion (RRF) combines both ranked lists using `1 / (RRF_K + rank)`.
4. An optional cross-encoder reranker can reorder the candidates.
5. The top chunks are inserted into a grounded prompt for Gemini.

RRF combines rank positions instead of adding raw semantic and BM25 scores, which are measured on different scales. This makes hybrid search simple and stable.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add `GEMINI_API_KEY` to `.env`. Place PDF files under `data/source_documents/Interface functions` or `data/source_documents/STEP functions`.

## Run indexing

```powershell
python -m rag.ingestion.build_index
```

## Ask questions

```powershell
python app.py
```

## Debug retrieval

```powershell
python -m rag.retrieval.debug "What encryption protocol is used?"
python -m rag.retrieval.debug "What encryption protocol is used?" --method bm25 --top-k 5
```

Debug output includes rank, score, retrieval source type, source file, page, section, chunk index, and the selection reason.

## Evaluate retrieval

Edit `evaluation/eval_questions.json`, then run:

```powershell
python -m evaluation.evaluate_retrieval
```

The summary compares semantic, BM25, and hybrid retrieval using Precision@K, Recall@K, F1@K, and MRR. Detailed results are written to `artifacts/reports/`.

## Tune retrieval

Edit `rag/config.py`:

- `CHUNK_SIZE` and `CHUNK_OVERLAP` control chunking. Rebuild the index after changing them.
- `TOP_K_SEMANTIC` and `TOP_K_BM25` control candidate counts.
- `FINAL_TOP_K` controls how many chunks reach the prompt.
- `RRF_K` controls how strongly RRF separates high and low ranks.
- `RERANK_ENABLED` enables optional cross-encoder reranking. It is `False` by default.

Change one parameter group at a time and compare evaluation metrics before deciding whether it helped.
