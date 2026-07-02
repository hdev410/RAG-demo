import pickle
from dataclasses import dataclass, field
from typing import Any, Literal

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import (
    BM25_INDEX_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    FINAL_TOP_K,
    RERANK_ENABLED,
    RRF_K,
    TOP_K_BM25,
    TOP_K_SEMANTIC,
)
from rag.retrieval.reranker import rerank
from rag.shared.text import tokenize

Method = Literal["semantic", "bm25", "hybrid"]


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    normalized["source_file"] = normalized.get(
        "source_file", normalized.get("source", "unknown")
    )
    normalized["page_number"] = normalized.get("page_number", normalized.get("page"))
    normalized["section_title"] = normalized.get(
        "section_title", normalized.get("heading", "unknown")
    )
    return normalized


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float
    source_type: str
    reason: str = ""
    component_ranks: dict[str, int] = field(default_factory=dict)


class Retriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        ).get_collection(COLLECTION_NAME)
        with BM25_INDEX_PATH.open("rb") as file:
            index = pickle.load(file)
        self.bm25, self.texts = index["bm25"], index["texts"]
        raw_metadata = index.get("metadata", index.get("metadatas"))
        self.metadata = [normalize_metadata(item) for item in raw_metadata]

    def semantic(self, query: str, group: str | None = None) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_embeddings=[self.model.encode(query).tolist()],
            n_results=TOP_K_SEMANTIC,
            where={"group": group} if group else None,
        )

        documents = result.get("documents")
        metadatas = result.get("metadatas")
        distances = result.get("distances")
        if not documents or not metadatas or not distances:
            return []

        return [
            RetrievedChunk(
                text=text,
                metadata=normalize_metadata(dict(metadata)),
                score=1 / (1 + distance),
                source_type="semantic",
                reason="High embedding similarity to the query",
            )
            for text, metadata, distance in zip(
                documents[0],
                metadatas[0],
                distances[0],
            )
        ]

    def lexical(self, query: str, group: str | None = None) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        results = []
        for index in ranked:
            metadata = self.metadata[index]
            if scores[index] <= 0 or (group and metadata.get("group") != group):
                continue
            results.append(
                RetrievedChunk(
                    self.texts[index],
                    metadata,
                    float(scores[index]),
                    "bm25",
                    "Strong exact keyword match",
                )
            )
            if len(results) == TOP_K_BM25:
                break
        return results

    @staticmethod
    def _key(chunk: RetrievedChunk) -> tuple:
        return (
            chunk.metadata.get("source_file"),
            chunk.metadata.get("page_number"),
            chunk.text,
        )

    def hybrid(self, query: str, group: str | None = None) -> list[RetrievedChunk]:
        fused: dict[tuple, RetrievedChunk] = {}

        # RRF uses rank positions instead of mixing incompatible raw score scales.
        for source_type, results in (
            ("semantic", self.semantic(query, group)),
            ("bm25", self.lexical(query, group)),
        ):
            for rank, chunk in enumerate(results, start=1):
                key = self._key(chunk)
                if key not in fused:
                    fused[key] = RetrievedChunk(
                        chunk.text,
                        chunk.metadata,
                        0.0,
                        "hybrid",
                        "Selected by RRF from semantic and/or BM25 ranking",
                    )
                fused[key].score += 1 / (RRF_K + rank)
                fused[key].component_ranks[source_type] = rank

        return sorted(fused.values(), key=lambda item: item.score, reverse=True)

    def retrieve(
        self,
        query: str,
        method: Method = "hybrid",
        group: str | None = None,
        top_k: int = FINAL_TOP_K,
    ) -> list[RetrievedChunk]:
        searches = {
            "semantic": self.semantic,
            "bm25": self.lexical,
            "hybrid": self.hybrid,
        }
        candidates = searches[method](query, group)
        return rerank(query, candidates, top_k, enabled=RERANK_ENABLED)
