from dotenv import load_dotenv

from rag.generation.answer import generate
from rag.retrieval.hybrid import Retriever


def main() -> None:
    load_dotenv()
    retriever = Retriever()
    print("RAG ready. Type 'exit' to quit.")
    while (question := input("\nQuestion: ").strip()).lower() not in {"exit", "quit"}:
        if not question:
            continue
        chunks = retriever.retrieve(question)
        print(f"\n{generate(question, chunks)}")
        print("\nSources:")
        for chunk in chunks:
            print(f"- {chunk.metadata.get('source')} | page {chunk.metadata.get('page')} | score {chunk.score:.4f}")


if __name__ == "__main__":
    main()
