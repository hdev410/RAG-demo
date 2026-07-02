import os

from google import genai

from rag.retrieval.hybrid import RetrievedChunk

SYSTEM_PROMPT = """You answer questions using only the supplied context.
If evidence is insufficient, say so. Keep the answer concise and cite source and page."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(
        f"Source: {chunk.metadata.get('source', 'unknown')}\n"
        f"Page: {chunk.metadata.get('page', 'unknown')}\n"
        f"Heading: {chunk.metadata.get('heading', 'unknown')}\n{chunk.text}"
        for chunk in chunks
    )


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context(chunks)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"GEMINI_API_KEY is not set. Retrieved context:\n\n{context}"
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
        )
        return response.text or "The model returned an empty response."
    except Exception as error:
        return f"Gemini request failed: {error}\n\nRetrieved context:\n\n{context}"
    finally:
        client.close()
