import re

from text_utils import clean_text, is_probable_heading, sliding_chunks
from config import STEP_CHUNK_SIZE, STEP_CHUNK_OVERLAP


TABLE_KEYWORDS = [
    "field",
    "field name",
    "control type",
    "mandatory",
    "editable",
    "default value",
    "description",
    "validation",
    "business rule",
    "data type",
    "length",
    "table",
    "column",
]


def looks_like_table_or_field_block(text: str) -> bool:
    lower = text.lower()
    keyword_hits = sum(1 for keyword in TABLE_KEYWORDS if keyword in lower)

    has_many_separators = lower.count(":") >= 3 or lower.count("|") >= 2
    has_field_pattern = bool(re.search(r"\b[A-Za-z0-9_]+Id\b", text))

    return keyword_hits >= 2 or has_many_separators or has_field_pattern


def split_page_into_blocks(text: str) -> list[str]:
    text = clean_text(text)

    if not text:
        return []

    blocks = re.split(r"\n\s*\n", text)

    cleaned_blocks = []

    for block in blocks:
        block = clean_text(block)

        if len(block) >= 40:
            cleaned_blocks.append(block)

    return cleaned_blocks


def chunk_step_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    current_heading = "General"

    for page in pages:
        lines = page["text"].splitlines()

        for line in lines:
            stripped = line.strip()

            if is_probable_heading(stripped):
                current_heading = stripped

        blocks = split_page_into_blocks(page["text"])

        for block_index, block in enumerate(blocks):
            if looks_like_table_or_field_block(block):
                chunk_type = "field_table"
            else:
                chunk_type = "page_section"

            sub_chunks = sliding_chunks(
                block,
                STEP_CHUNK_SIZE,
                STEP_CHUNK_OVERLAP
            )

            for sub_index, sub_text in enumerate(sub_chunks):
                chunks.append({
                    "text": (
                        f"Heading: {current_heading}\n"
                        f"Page: {page['page']}\n"
                        f"Content Type: {chunk_type}\n\n"
                        f"{sub_text}"
                    ),
                    "metadata": {
                        "group": "step",
                        "source": page["source"],
                        "page": page["page"],
                        "heading": current_heading,
                        "chunk_type": chunk_type,
                        "block_index": block_index,
                        "chunk_index": sub_index
                    }
                })

    return chunks