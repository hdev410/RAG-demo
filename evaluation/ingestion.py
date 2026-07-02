import csv
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
import fitz

from rag.config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, ROOT
from rag.ingestion.loader import document_group

REPORT_PATH = ROOT / "artifacts" / "reports" / "ingestion_audit.csv"


def count_pdf_pages_and_text(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)

    page_count = len(doc)
    total_chars = 0
    empty_pages = 0

    for page in doc:
        text = str(page.get_text("text") or "")
        chars = len(text.strip())
        total_chars += chars

        if chars == 0:
            empty_pages += 1

    return {
        "source": pdf_path.name,
        "path": str(pdf_path),
        "page_count": page_count,
        "raw_text_chars": total_chars,
        "empty_pages": empty_pages,
    }


def get_chunk_stats() -> dict:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    result = collection.get(include=["documents", "metadatas"])

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    chunk_count = Counter()
    chunk_chars = defaultdict(int)

    for doc, metadata in zip(documents, metadatas):
        source = metadata.get("source_file", metadata.get("source", "unknown"))
        chunk_count[source] += 1
        chunk_chars[source] += len(str(doc))

    return {
        "chunk_count": chunk_count,
        "chunk_chars": chunk_chars,
    }


def main():
    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    chunk_stats = get_chunk_stats()

    rows = []

    for pdf_path in pdf_files:
        pdf_info = count_pdf_pages_and_text(pdf_path)
        source = pdf_info["source"]

        chunk_count = chunk_stats["chunk_count"].get(source, 0)
        chunk_chars = chunk_stats["chunk_chars"].get(source, 0)

        raw_text_chars = pdf_info["raw_text_chars"]

        coverage_ratio = (
            round(chunk_chars / raw_text_chars, 3) if raw_text_chars > 0 else 0
        )

        avg_chunk_chars = round(chunk_chars / chunk_count, 1) if chunk_count > 0 else 0

        warning = ""

        if raw_text_chars == 0:
            warning = "NO_TEXT_EXTRACTED"
        elif chunk_count == 0:
            warning = "NO_CHUNKS_IN_DB"
        elif document_group(pdf_path) == "interface" and coverage_ratio < 0.25:
            warning = "LOW_COVERAGE"
        elif document_group(pdf_path) == "step" and coverage_ratio < 0.45:
            warning = "LOW_COVERAGE"
        elif pdf_info["empty_pages"] > pdf_info["page_count"] * 0.3:
            warning = "MANY_EMPTY_PAGES"

        rows.append(
            {
                "group": document_group(pdf_path),
                "source": source,
                "page_count": pdf_info["page_count"],
                "raw_text_chars": raw_text_chars,
                "empty_pages": pdf_info["empty_pages"],
                "chunk_count": chunk_count,
                "chunk_chars": chunk_chars,
                "avg_chunk_chars": avg_chunk_chars,
                "coverage_ratio": coverage_ratio,
                "warning": warning,
            }
        )

    rows.sort(key=lambda row: (row["warning"] == "", row["group"], row["source"]))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "group",
                "source",
                "page_count",
                "raw_text_chars",
                "empty_pages",
                "chunk_count",
                "chunk_chars",
                "avg_chunk_chars",
                "coverage_ratio",
                "warning",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Audit report saved to: {REPORT_PATH}")

    warnings = [row for row in rows if row["warning"]]

    print(f"Total PDFs: {len(rows)}")
    print(f"PDFs with warnings: {len(warnings)}")

    for row in warnings[:20]:
        print(
            f"{row['warning']:18} | "
            f"{row['group']:10} | "
            f"pages={row['page_count']:4} | "
            f"chunks={row['chunk_count']:5} | "
            f"coverage={row['coverage_ratio']} | "
            f"{row['source']}"
        )


if __name__ == "__main__":
    main()
