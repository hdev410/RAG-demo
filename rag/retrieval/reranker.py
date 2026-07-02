from typing import TYPE_CHECKING

from rag.config import RERANK_MODEL

if TYPE_CHECKING:
    from rag.retrieval.hybrid import RetrievedChunk


def rerank(
    query: str,
    chunks: list["RetrievedChunk"],
    top_k: int,
    enabled: bool = False,
) -> list["RetrievedChunk"]:
    if not enabled or not chunks:
        return chunks[:top_k]

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(RERANK_MODEL)
        scores = model.predict([(query, chunk.text) for chunk in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
            chunk.source_type = "reranker"
            chunk.reason = "Cross-encoder relevance score"
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:top_k]
    except Exception as error:
        print(f"Reranker unavailable ({error}); using original ranking.")
        return chunks[:top_k]
