import argparse

from rag.config import FINAL_TOP_K
from rag.retrieval.hybrid import Method, Retriever


def print_results(query: str, method: Method = "hybrid", top_k: int = FINAL_TOP_K) -> None:
    """Print ranking details to explain why each chunk was retrieved."""
    chunks = Retriever().retrieve(query, method=method, top_k=top_k)
    print(f"\nQuery: {query}\nMethod: {method}\nResults: {len(chunks)}")

    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        ranks = ", ".join(f"{name}=#{value}" for name, value in chunk.component_ranks.items())
        print("\n" + "=" * 80)
        print(f"Rank: {rank} | Score: {chunk.score:.6f} | Source type: {chunk.source_type}")
        print(f"Source: {metadata.get('source_file', 'unknown')}")
        print(f"Page: {metadata.get('page_number', 'unknown')}")
        print(f"Section: {metadata.get('section_title', 'unknown')}")
        print(f"Chunk index: {metadata.get('chunk_index', 'unknown')}")
        print(f"Why: {chunk.reason}" + (f" ({ranks})" if ranks else ""))
        print(f"\n{chunk.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect retrieved RAG chunks")
    parser.add_argument("query", help="Question to retrieve context for")
    parser.add_argument("--method", choices=("semantic", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    args = parser.parse_args()
    print_results(args.query, args.method, args.top_k)


if __name__ == "__main__":
    main()
