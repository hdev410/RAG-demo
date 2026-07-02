import json

from rag.config import ROOT
from rag.retrieval.hybrid import Method, Retriever

QUESTIONS = ROOT / "data" / "ground_truth" / "questions.json"
REPORT = ROOT / "artifacts" / "reports" / "retrieval_comparison.json"


def normalize(value: str) -> list[str]:
    ignored = {"functional", "specification", "interface", "final", "signed", "completed", "pdf"}
    words = value.lower().replace("_", " ").replace("-", " ").split()
    return [word.strip("().") for word in words if len(word.strip("().")) > 2 and word.strip("().") not in ignored]


def source_matches(actual: str, expected: str) -> bool:
    actual_terms = set(normalize(actual))
    return set(normalize(expected)).issubset(actual_terms)


def evaluate(retriever: Retriever, samples: list[dict], method: Method) -> dict:
    rows, reciprocal_ranks = [], []
    for sample in samples:
        chunks = retriever.retrieve(sample["question"], method=method)
        rank = next((i for i, chunk in enumerate(chunks, 1) if source_matches(str(chunk.metadata.get("source", "")), sample["expected_source_contains"])), None)
        reciprocal_ranks.append(1 / rank if rank else 0)
        rows.append({"question": sample["question"], "first_relevant_rank": rank, "sources": [chunk.metadata.get("source") for chunk in chunks]})
    return {"method": method, "hit_rate": sum(rank > 0 for rank in reciprocal_ranks) / len(rows), "mrr": sum(reciprocal_ranks) / len(rows), "results": rows}


def main() -> None:
    samples = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = Retriever()
    reports = [evaluate(retriever, samples, method) for method in ("semantic", "bm25", "hybrid")]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    for report in reports:
        print(f"{report['method']:8} | Hit@3 {report['hit_rate']:.2%} | MRR {report['mrr']:.4f}")
    print(f"Saved: {REPORT}")


if __name__ == "__main__":
    main()
