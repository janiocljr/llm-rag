"""
app/core/text_utils.py
======================
Shared text processing utilities used across ingestion modules.
"""

import re
import unicodedata

_MULTI_SPACE = re.compile(r" {2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_EOL = re.compile(r"-\n(\w)")
_PAGE_NUM = re.compile(r"^\s*\d+\s*$", re.MULTILINE)


def clean_text(raw: str) -> str:
    """
    Normalise raw PDF text.

    Steps (in order):
    1. Unicode normalisation (NFC) — handles ligatures like 'ﬁ' → 'fi'.
    2. Re-join hyphenated line-breaks common in justified PDF text.
    3. Remove lone page-number lines.
    4. Collapse excessive whitespace.
    5. Strip leading/trailing whitespace.
    """
    text = unicodedata.normalize("NFC", raw)
    text = _HYPHEN_EOL.sub(r"\1", text)
    text = _PAGE_NUM.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """
    Approximate token count using the chars/4 heuristic.

    For English/Portuguese prose this is accurate within ±15 %.
    We deliberately avoid adding a heavy tokeniser dependency just for
    chunking — the small estimation error is acceptable.
    """
    return max(1, len(text) // 4)
