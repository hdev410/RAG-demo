import json
from dataclasses import dataclass

from rag.config import FINAL_TOP_K, ROOT
from rag.retrieval.hybrid import Method, RetrievedChunk, Retriever

EVAL_FILE = ROOT / "evaluation" / "eval_questions.json"
REPORT_FILE = ROOT / "artifacts" / "reports" / "retrieval_evaluation.json"
PAGE_TOLERANCE = 2  # PDF page labels can differ slightly from extracted page indexes.


@dataclass
class Metrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    reciprocal_rank: float = 0.0


def normalize(value: str) -> set[str]:
    ignored = {
        "functional",
        "specification",
        "interface",
        "final",
        "signed",
        "completed",
        "pdf",
    }
    words = value.lower().replace("_", " ").replace("-", " ").split()
    return {word.strip("().") for word in words if len(word.strip("().")) > 2} - ignored


def source_matches(actual: str, expected: str) -> bool:
    return normalize(expected).issubset(normalize(actual))


def matched_page(chunk: RetrievedChunk, sample: dict) -> int | None:
    if not source_matches(
        str(chunk.metadata.get("source_file", "")), sample["expected_source_contains"]
    ):
        return None

    expected_pages = sample.get("expected_pages", [])
    if not expected_pages:
        return 0

    page_value = chunk.metadata.get("page_number")
    if not isinstance(page_value, (int, str)):
        return None

    try:
        actual_page = int(page_value)
    except ValueError:
        return None
    return next(
        (page for page in expected_pages if abs(actual_page - page) <= PAGE_TOLERANCE),
        None,
    )


def score_question(chunks: list[RetrievedChunk], sample: dict, top_k: int) -> Metrics:
    matches = [matched_page(chunk, sample) for chunk in chunks]
    relevant_ranks = [
        rank for rank, match in enumerate(matches, start=1) if match is not None
    ]
    expected_count = max(1, len(sample.get("expected_pages", [])))

    # Precision counts relevant chunks; recall counts distinct labeled pages found.
    precision = len(relevant_ranks) / top_k
    recall = min(
        1.0, len({match for match in matches if match is not None}) / expected_count
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    return Metrics(precision, recall, f1, reciprocal_rank)


def evaluate(retriever: Retriever, samples: list[dict], method: Method) -> dict:
    rows = []
    for sample in samples:
        chunks = retriever.retrieve(
            sample["question"], method=method, top_k=FINAL_TOP_K
        )
        metrics = score_question(chunks, sample, FINAL_TOP_K)
        rows.append(
            {
                "question": sample["question"],
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "reciprocal_rank": metrics.reciprocal_rank,
                "retrieved_sources": [
                    chunk.metadata.get("source_file") for chunk in chunks
                ],
            }
        )

    count = len(rows)
    return {
        "method": method,
        "precision": sum(row["precision"] for row in rows) / count,
        "recall": sum(row["recall"] for row in rows) / count,
        "f1": sum(row["f1"] for row in rows) / count,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
        "questions": rows,
    }


def print_summary(reports: list[dict]) -> None:
    print(f"\nRetrieval evaluation at K={FINAL_TOP_K}")
    print("+----------+-------------+----------+----------+----------+")
    print("| Method   | Precision@K | Recall@K | F1@K     | MRR      |")
    print("+----------+-------------+----------+----------+----------+")
    for report in reports:
        print(
            f"| {report['method']:<8} | {report['precision']:<11.4f} | "
            f"{report['recall']:<8.4f} | {report['f1']:<8.4f} | {report['mrr']:<8.4f} |"
        )
    print("+----------+-------------+----------+----------+----------+")


def main() -> None:
    samples = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    retriever = Retriever()
    reports = [
        evaluate(retriever, samples, method)
        for method in ("semantic", "bm25", "hybrid")
    ]
    print_summary(reports)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Detailed report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
