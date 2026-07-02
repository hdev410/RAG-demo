# RAG Demo

This project demonstrates a small, complete Retrieval-Augmented Generation pipeline: PDF ingestion, heading-aware chunking, semantic/BM25 retrieval, rank fusion, evaluation, and Gemini answer generation.

## Structure

```text
rag_project/
├── app.py                         # Interactive RAG CLI
├── rag/
│   ├── config.py                  # Paths and shared settings
│   ├── ingestion/
│   │   ├── loader.py              # Extract PDF pages
│   │   ├── chunking.py            # Build heading-aware chunks
│   │   └── build_index.py         # Create Chroma and BM25 indexes
│   ├── retrieval/
│   │   └── hybrid.py              # Semantic, BM25, and RRF retrieval
│   ├── generation/
│   │   └── answer.py              # Build context and call Gemini
│   └── shared/
│       └── text.py                # Text cleaning and tokenization
├── evaluation/
│   ├── retrieval.py               # Compare retrieval methods
│   ├── ingestion.py               # Audit PDF/chunk coverage
│   └── inspect_index.py            # Preview indexed chunks
├── data/
│   ├── source_documents/          # PDF input (not committed)
│   └── ground_truth/questions.json
└── artifacts/
    ├── indexes/                    # Generated Chroma/BM25 data
    └── reports/                    # Generated evaluation reports
```

## Commands

```powershell
python -m rag.ingestion.build_index
python app.py
python -m evaluation.retrieval
python -m evaluation.ingestion
python -m evaluation.inspect_index
```

Retrieval modes are `semantic` (meaning), `bm25` (exact terms), and `hybrid` (RRF combines both rankings).