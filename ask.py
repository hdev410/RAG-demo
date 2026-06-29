import os

from dotenv import load_dotenv
from google import genai

from retriever import HybridRetriever


SYSTEM_PROMPT = """
You are a RAG assistant for system functional specification documents.

Rules:
1. Answer only using the provided context.
2. If the context is not enough, say you cannot find enough information.
3. Always mention source file and page when possible.
4. Be concise but specific.
"""


def build_context(chunks):
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        heading = metadata.get("heading", "unknown")

        context_parts.append(
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Heading: {heading}\n"
            f"Text:\n{chunk.text}"
        )

    return "\n\n---\n\n".join(context_parts)


def answer_with_gemini(question: str, context: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return f"No GEMINI_API_KEY found. Showing retrieved context only:\n\n{context}"

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}
"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text

    except Exception as error:
        return (
            "Gemini call failed. Showing retrieved context only.\n\n"
            f"Reason: {error}\n\n"
            f"{context}"
        )


def main():
    load_dotenv()

    retriever = HybridRetriever()

    print("RAG is ready. Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        chunks = retriever.retrieve(question)
        context = build_context(chunks)

        answer = answer_with_gemini(question, context)

        print("\nAnswer:")
        print(answer)

        print("\nSources:")
        for chunk in chunks:
            metadata = chunk.metadata
            print(
                f"- {metadata.get('source')} | "
                f"page {metadata.get('page')} | "
                f"{metadata.get('heading')}"
            )


if __name__ == "__main__":
    main()