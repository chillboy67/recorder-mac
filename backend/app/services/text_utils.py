import re


WORD_RE = re.compile(r"[A-Za-z']+")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

