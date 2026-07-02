import os

from google import genai

from rag.retrieval.hybrid import RetrievedChunk

SYSTEM_PROMPT = '''Answer using only the provided context.
Do not add facts from outside knowledge.
Cite the source file and page number for factual claims.
If the context is insufficient, reply: "The answer was not found in the provided documents."'''


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Turn retrieved chunks and their citations into the LLM context block."""
    return "\n\n---\n\n".join(
        f"Source file: {chunk.metadata.get('source_file', 'unknown')}\n"
        f"Page number: {chunk.metadata.get('page_number', 'unknown')}\n"
        f"Section: {chunk.metadata.get('section_title', 'unknown')}\n{chunk.text}"
        for chunk in chunks
    )


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build a grounded prompt and ask Gemini to answer from context only."""
    context = build_context(chunks)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"GEMINI_API_KEY is not set. Retrieved context:\n\n{context}"
    # The prompt clearly separates instructions, evidence, and the user question.
    prompt = f"{SYSTEM_PROMPT}\n\nRetrieved context:\n{context}\n\nQuestion: {question}\nAnswer:"
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
