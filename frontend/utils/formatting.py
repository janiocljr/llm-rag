from __future__ import annotations


def truncate(text: str, max_words: int = 40) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def score_color(score: float) -> str:
    if score >= 0.75:
        return "score-high"
    if score >= 0.50:
        return "score-mid"
    return "score-low"


def format_latency(latency_ms: float) -> str:
    if latency_ms < 1_000:
        return f"{latency_ms:.0f} ms"
    seconds = latency_ms / 1_000
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes} min {secs} s"
