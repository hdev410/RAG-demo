# RAG Demo Structure

This folder contains a small PDF RAG pipeline with offline indexing, hybrid retrieval, optional reranking, evaluation, and Gemini answer generation.

```text
rag_project/
|-- app.py                              Interactive query CLI
|-- rag/
|   |-- config.py                       Paths and tunable parameters
|   |-- ingestion/
|   |   |-- loader.py                   Extract PDF pages
|   |   |-- chunking.py                 Create heading-aware chunks
|   |   `-- build_index.py              Build ChromaDB and BM25 indexes
|   |-- retrieval/
|   |   |-- hybrid.py                   Semantic, BM25, and RRF retrieval
|   |   |-- reranker.py                 Optional second-stage reranking
|   |   `-- debug.py                    Inspect retrieved chunks
|   |-- generation/answer.py            Build prompt and call Gemini
|   `-- shared/text.py                  Text cleaning and tokenization
|-- evaluation/
|   |-- eval_questions.json             Retrieval ground truth
|   |-- evaluate_retrieval.py           Precision, recall, F1, and MRR
|   |-- ingestion.py                    Audit document/chunk coverage
|   `-- inspect_index.py                Preview indexed chunks
|-- data/source_documents/              Input PDFs (not committed)
`-- artifacts/
    |-- indexes/                         Generated vector/BM25 indexes
    `-- reports/                         Generated evaluation reports
```

See `README.md` for setup, commands, retrieval concepts, and tuning guidance.
