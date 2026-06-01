"""Pipeline diagram — generates and renders an SVG of the RAG flow."""
from __future__ import annotations

import streamlit as st

# ── Node definitions ──────────────────────────────────────────────────────────
_NODES = [
    ("PDF", 60,  110, "#2d8cf0"),
    ("Chunking", 190, 110, "#9b59b6"),
    ("Embedder", 320, 110, "#27ae60"),
    ("FAISS\nIndex", 450, 110, "#e67e22"),
    ("MMR\nRe-rank", 580, 110, "#e74c3c"),
    ("LLM", 710, 110, "#1abc9c"),
    ("Answer", 840, 110, "#f39c12"),
]

_W, _H = 940, 220
_R = 36   # node radius


def _build_svg() -> str:
    """Build the RAG pipeline SVG string."""
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}" style="font-family:sans-serif;">'
    ]

    # Background
    parts.append(
        f'<rect width="{_W}" height="{_H}" rx="12" fill="#0d1117"/>'
    )

    # Arrows between nodes
    for i in range(len(_NODES) - 1):
        _, x1, y, _ = _NODES[i]
        _, x2, _, _ = _NODES[i + 1]
        ax1 = x1 + _R
        ax2 = x2 - _R
        parts.append(
            f'<line x1="{ax1}" y1="{y}" x2="{ax2}" y2="{y}" '
            f'stroke="#4a5568" stroke-width="2" marker-end="url(#arrow)"/>'
        )

    # Arrow marker
    parts.append(
        '<defs>'
        '<marker id="arrow" markerWidth="8" markerHeight="8" '
        'refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L8,3 z" fill="#4a5568"/>'
        '</marker>'
        '</defs>'
    )

    # Nodes
    for label, cx, cy, color in _NODES:
        lines = label.split("\n")
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{_R}" fill="{color}" '
            f'fill-opacity="0.25" stroke="{color}" stroke-width="2"/>'
        )
        offset = -8 * (len(lines) - 1) / 2
        for i, line in enumerate(lines):
            dy = cy + offset + i * 16
            parts.append(
                f'<text x="{cx}" y="{dy}" text-anchor="middle" '
                f'fill="#e2e8f0" font-size="11" font-weight="600">{line}</text>'
            )

    # Step labels below
    step_labels = ["1. Ingestão", "2. Chunk", "3. Embed", "4. Busca", "5. MMR", "6. Gera", "7. Resp."]
    for (_, cx, cy, _), lbl in zip(_NODES, step_labels):
        parts.append(
            f'<text x="{cx}" y="{cy + _R + 18}" text-anchor="middle" '
            f'fill="#718096" font-size="10">{lbl}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def render_pipeline_diagram() -> None:
    """Render the RAG pipeline SVG inside a styled container."""
    with st.expander("🔄 Diagrama do Pipeline RAG", expanded=False):
        svg = _build_svg()
        st.markdown(
            f'<div class="pipeline-wrap">{svg}</div>',
            unsafe_allow_html=True,
        )
