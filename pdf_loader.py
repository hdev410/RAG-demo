import fitz
from pathlib import Path


def detect_group(file_path: Path) -> str:
    path_text = str(file_path).lower()

    if "interface functions" in path_text:
        return "interface"

    if "step functions" in path_text:
        return "step"

    return "unknown"


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []

    for index, page in enumerate(doc):
        text = page.get_text("text")

        pages.append({
            "source": pdf_path.name,
            "path": str(pdf_path),
            "group": detect_group(pdf_path),
            "page": index + 1,
            "text": text or ""
        })

    return pages


def load_all_pdfs(data_dir: Path) -> list[dict]:
    all_pages = []

    pdf_files = list(data_dir.rglob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        pages = load_pdf_pages(pdf_path)
        all_pages.extend(pages)

    return all_pages