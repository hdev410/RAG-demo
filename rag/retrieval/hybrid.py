import pickle
from dataclasses import dataclass
from typing import Any, Literal

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import BM25_INDEX_PATH, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, FINAL_K, RETRIEVAL_K, RRF_K
from rag.shared.text import tokenize

Method = Literal["semantic", "bm25", "hybrid"]


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float
    method: str


class Retriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
        with BM25_INDEX_PATH.open("rb") as file:
            index = pickle.load(file)
        self.bm25, self.texts = index["bm25"], index["texts"]
        self.metadata = index.get("metadata", index.get("metadatas"))
        self.ids = index["ids"]

    def semantic(self, query: str, group: str | None = None) -> list[RetrievedChunk]:
        # Semantic search compares embedding distance and handles paraphrases well.
        result = self.collection.query(
            query_embeddings=[self.model.encode(query).tolist()],
            n_results=RETRIEVAL_K,
            where={"group": group} if group else None,
        )
        return [
            RetrievedChunk(text, dict(meta), 1 / (1 + distance), "semantic")
            for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    def lexical(self, query: str, group: str | None = None) -> list[RetrievedChunk]:
        # BM25 compares exact terms and is strong for IDs, names, and rare keywords.
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        results = []
        for index in ranked:
            if scores[index] <= 0 or (group and self.metadata[index].get("group") != group):
                continue
            results.append(RetrievedChunk(self.texts[index], dict(self.metadata[index]), float(scores[index]), "bm25"))
            if len(results) == RETRIEVAL_K:
                break
        return results

    @staticmethod
    def _key(chunk: RetrievedChunk) -> tuple:
        return (chunk.metadata.get("source"), chunk.metadata.get("page"), chunk.text)

    def hybrid(self, query: str, group: str | None = None) -> list[RetrievedChunk]:
        # RRF combines ranks, avoiding incompatible cosine and BM25 score scales.
        fused: dict[tuple, RetrievedChunk] = {}
        for results in (self.semantic(query, group), self.lexical(query, group)):
            for rank, chunk in enumerate(results, start=1):
                key = self._key(chunk)
                if key not in fused:
                    fused[key] = RetrievedChunk(chunk.text, chunk.metadata, 0.0, "hybrid")
                fused[key].score += 1 / (RRF_K + rank)
        return sorted(fused.values(), key=lambda item: item.score, reverse=True)

    def retrieve(self, query: str, method: Method = "hybrid", group: str | None = None, top_k: int = FINAL_K) -> list[RetrievedChunk]:
        searches = {"semantic": self.semantic, "bm25": self.lexical, "hybrid": self.hybrid}
        return searches[method](query, group)[:top_k]
