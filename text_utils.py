import re


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_search(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_./-]+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_for_search(text)
    return normalized.split()


def is_probable_heading(line: str) -> bool:
    line = line.strip()

    patterns = [
        r"^\d+(\.\d+)*\s+[A-Z].+",
        r"^[A-Z][A-Za-z ]{3,80}$",
        r"^Appendix\s+[A-Z0-9]+",
    ]

    return any(re.match(pattern, line) for pattern in patterns)


def sliding_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks