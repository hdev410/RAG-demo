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
    pages = []
    for path in sorted(data_dir.rglob("*.pdf")):
        with fitz.open(path) as document:
            for page_index in range(len(document)):
                page = document.load_page(page_index)
                pages.append(
                    {
                        "source_file": path.name,
                        "source_path": str(path),
                        "group": document_group(path),
                        "page_number": page_index + 1,
                        "text": page.get_text("text") or "",
                    }
                )
    return pages
