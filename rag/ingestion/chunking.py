from collections import defaultdict

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.shared.text import clean, is_heading, is_low_value, split_with_overlap


def _document_sections(pages: list[dict]) -> list[tuple[str, int, str]]:
    sections = []
    heading, start_page, lines = "General", pages[0]["page"], []

    def flush() -> None:
        nonlocal lines
        text = clean("\n".join(lines))
        if len(text) >= 80 and not is_low_value(heading, text):
            sections.append((heading, start_page, text))
        lines = []

    for page in pages:
        for line in page["text"].splitlines():
            if is_heading(line):
                flush()
                heading, start_page = line.strip(), page["page"]
            else:
                lines.append(line)
    flush()
    return sections


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Create heading-aware chunks while keeping source metadata."""
    documents = defaultdict(list)
    for page in pages:
        documents[(page["group"], page["source"])].append(page)

    chunks = []
    for (group, source), document_pages in documents.items():
        if group == "unknown":
            continue
        for heading, page, section in _document_sections(document_pages):
            for index, text in enumerate(split_with_overlap(section, CHUNK_SIZE, CHUNK_OVERLAP)):
                chunks.append(
                    {
                        "text": f"Heading: {heading}\n\n{text}",
                        "metadata": {
                            "group": group,
                            "source": source,
                            "page": page,
                            "heading": heading,
                            "chunk_index": index,
                        },
                    }
                )
    return chunks
