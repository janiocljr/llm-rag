import re
import unicodedata

_MULTI_SPACE = re.compile(r" {2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_EOL = re.compile(r"-\n(\w)")
_PAGE_NUM = re.compile(r"^\s*\d+\s*$", re.MULTILINE)


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = _HYPHEN_EOL.sub(r"\1", text)
    text = _PAGE_NUM.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
