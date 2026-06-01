"""Pure formatting helpers — no Streamlit, no I/O."""
from __future__ import annotations


def truncate(text: str, max_words: int = 40) -> str:
    """Truncate text to at most *max_words* words, appending '…' if cut."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def score_color(score: float) -> str:
    """Return the CSS class name for a similarity score badge.

    Returns one of: 'score-high', 'score-mid', 'score-low'.
    """
    if score >= 0.75:
        return "score-high"
    if score >= 0.50:
        return "score-mid"
    return "score-low"


def format_latency(latency_ms: float) -> str:
    """Human-readable latency string.

    Examples:
        342.0  → '342 ms'
        1500.0 → '1.5 s'
        75000  → '1 min 15 s'
    """
    if latency_ms < 1_000:
        return f"{latency_ms:.0f} ms"
    seconds = latency_ms / 1_000
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes} min {secs} s"
