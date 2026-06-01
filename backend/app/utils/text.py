"""Text cleaning and tokenization utilities."""
from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize unicode, remove control characters, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newline and tab
    text = re.sub(r"[^\S\n\t ]+", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"
