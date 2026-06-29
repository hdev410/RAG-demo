import pickle
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    BM25_INDEX_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K_SEMANTIC,
    TOP_K_BM25,
    FINAL_TOP_K,
)
from text_utils import tokenize


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float
    source_type: str


STEP_KEYWORDS = [
    "field",
    "fields",
    "mandatory",
    "optional",
    "control type",
    "editable",
    "default value",
    "validation",
    "table",
    "column",
    "description",
    "data type",
    "length",
]

INTERFACE_KEYWORDS = [
    "feature",
    "screen",
    "ui",
    "web app",
    "interface",
    "api",
    "flow",
    "overview",
    "authentication",
    "integration",
]


def route_query(question: str) -> str:
    lower = question.lower()

    if any(keyword in lower for keyword in STEP_KEYWORDS):
        return "step"

    if any(keyword in lower for keyword in INTERFACE_KEYWORDS):
        return "interface"

    return "all"


class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

        with open(BM25_INDEX_PATH, "rb") as file:
            bm25_data = pickle.load(file)

        self.bm25 = bm25_data["bm25"]
        self.bm25_texts = bm25_data["texts"]
        self.bm25_metadatas = bm25_data["metadatas"]
        self.bm25_ids = bm25_data["ids"]

    def semantic_search(self, question: str, group: str) -> list[RetrievedChunk]:
        query_embedding = self.embedding_model.encode(question).tolist()

        where_filter = None

        if group != "all":
            where_filter = {"group": group}

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_SEMANTIC,
            where=where_filter,
        )

        chunks = []

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            score = 1 / (1 + distance)

            chunks.append(
                RetrievedChunk(
                    text=text, metadata=metadata, score=score, source_type="semantic"
                )
            )

        return chunks

    def bm25_search(self, question: str, group: str) -> list[RetrievedChunk]:
        query_tokens = tokenize(question)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )

        chunks = []

        for index in ranked_indexes:
            metadata = self.bm25_metadatas[index]

            if group != "all" and metadata.get("group") != group:
                continue

            if scores[index] <= 0:
                continue

            chunks.append(
                RetrievedChunk(
                    text=self.bm25_texts[index],
                    metadata=metadata,
                    score=float(scores[index]),
                    source_type="bm25",
                )
            )

            if len(chunks) >= TOP_K_BM25:
                break

        return chunks

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        group = route_query(question)

        semantic_results = self.semantic_search(question, group)
        bm25_results = self.bm25_search(question, group)

        merged = {}

        for chunk in semantic_results + bm25_results:
            key = (
                chunk.metadata.get("source"),
                chunk.metadata.get("page"),
                chunk.metadata.get("heading"),
                chunk.text[:100],
            )

            if key not in merged:
                merged[key] = chunk
            else:
                merged[key].score += chunk.score
        question_lower = question.lower()

        for chunk in merged.values():
            source_lower = str(chunk.metadata.get("source", "")).lower()

            if "np websites" in question_lower:
                if "np websites" in source_lower:
                    chunk.score += 10
                elif "nyp websites" in source_lower or "np common app" in source_lower:
                    chunk.score -= 15
        final_results = sorted(
            merged.values(), key=lambda chunk: chunk.score, reverse=True
        )

        return final_results[:FINAL_TOP_K]
