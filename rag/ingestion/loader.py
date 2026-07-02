from pathlib import Path

import fitz


def document_group(path: Path) -> str:
    value = str(path).lower()
    if "interface functions" in value:
        return "interface"
    if "step functions" in value:
        return "step"
    return "unknown"


def load_pdfs(data_dir: Path) -> list[dict]:
    """Extract text page by page so each chunk can keep its PDF page number."""
    pages = []
    for path in sorted(data_dir.rglob("*.pdf")):
        with fitz.open(path) as document:
            pages.extend(
                {
                    "source_file": path.name,
                    "source_path": str(path),
                    "group": document_group(path),
                    "page_number": number,
                    "text": page.get_text("text") or "",
                }
                for number, page in enumerate(document, start=1)
            )
    return pages
