from text_utils import clean_text, is_probable_heading, sliding_chunks
from config import INTERFACE_CHUNK_SIZE, INTERFACE_CHUNK_OVERLAP


def chunk_interface_pages(pages: list[dict]) -> list[dict]:
    chunks = []

    current_heading = "General"
    current_text = []
    current_page = None
    current_source = None

    def flush():
        nonlocal current_text, current_heading, current_page, current_source

        text = clean_text("\n".join(current_text))

        if len(text) < 80:
            current_text = []
            return

        sub_chunks = sliding_chunks(
            text,
            INTERFACE_CHUNK_SIZE,
            INTERFACE_CHUNK_OVERLAP
        )

        for index, sub_text in enumerate(sub_chunks):
            chunks.append({
                "text": f"Heading: {current_heading}\n\n{sub_text}",
                "metadata": {
                    "group": "interface",
                    "source": current_source,
                    "page": current_page,
                    "heading": current_heading,
                    "chunk_type": "section",
                    "chunk_index": index
                }
            })

        current_text = []

    for page in pages:
        current_source = page["source"]
        lines = page["text"].splitlines()

        for line in lines:
            stripped = line.strip()

            if not stripped:
                current_text.append("")
                continue

            if is_probable_heading(stripped):
                flush()
                current_heading = stripped
                current_page = page["page"]
            else:
                if current_page is None:
                    current_page = page["page"]

                current_text.append(stripped)

    flush()

    return chunks