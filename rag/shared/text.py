import re

LOW_VALUE_MARKERS = (
    "table of contents",
    "version history",
    "revision history",
    "document acceptance form",
    "signature",
)


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9_./-]+", " ", text.lower()).split()


def is_low_value(heading: str, text: str) -> bool:
    value = f"{heading} {text}".lower()
    return any(marker in value for marker in LOW_VALUE_MARKERS)


def is_heading(line: str) -> bool:
    value = line.strip()
    patterns = (
        r"^\d+(\.\d+)*\s+[A-Z].+",
        r"^[A-Z][A-Za-z ]{3,80}$",
        r"^Appendix\s+[A-Z0-9]+",
    )
    return any(re.match(pattern, value) for pattern in patterns)


def split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    if size <= overlap:
        raise ValueError("Chunk size must be greater than overlap")
    return [text[start : start + size].strip() for start in range(0, len(text), size - overlap) if text[start : start + size].strip()]
