from __future__ import annotations

import streamlit as st


_NODES: list[tuple[list[str], str, str]] = [
    (["PDF"], "1 · Ingestão", "#6366F1"),
    (["Chunking"], "2 · Divisão", "#6366F1"),
    (["Embeddings"], "3 · Vetorização", "#8B5CF6"),
    (["Índice", "FAISS"], "4 · Busca", "#8B5CF6"),
    (["Re-rank", "MMR"], "5 · Diversidade", "#8B5CF6"),
    (["LLM"], "6 · Geração", "#22C55E"),
    (["Resposta"], "7 · Citações", "#22C55E"),
]

_NODE_W, _NODE_H = 104, 46
_GAP = 36
_TOP = 16
_W = len(_NODES) * _NODE_W + (len(_NODES) - 1) * _GAP + 8
_H = 104


def _build_svg() -> str:
    cy = _TOP + _NODE_H / 2
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'role="img" aria-label="Pipeline RAG: PDF, chunking, embeddings, '
        f'índice FAISS, re-rank MMR, LLM e resposta" '
        f'style="font-family:Inter,sans-serif;">',
        '<defs>'
        '<marker id="arrow" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto">'
        '<path d="M0,0 L0,5 L6,2.5 z" fill="#3F3F46"/>'
        '</marker>'
        '</defs>',
    ]

    for i in range(len(_NODES) - 1):
        x1 = 4 + (i + 1) * _NODE_W + i * _GAP + 6
        x2 = x1 + _GAP - 12
        parts.append(
            f'<line x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}" '
            f'stroke="#3F3F46" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )

    for i, (lines, caption, color) in enumerate(_NODES):
        x = 4 + i * (_NODE_W + _GAP)
        cx = x + _NODE_W / 2
        parts.append(
            f'<rect x="{x}" y="{_TOP}" width="{_NODE_W}" height="{_NODE_H}" rx="10" '
            f'fill="#131316" stroke="#26262B" stroke-width="1"/>'
        )
        parts.append(f'<circle cx="{x + 14}" cy="{_TOP + 14}" r="3" fill="{color}"/>')

        offset = -6.5 * (len(lines) - 1)
        for j, line in enumerate(lines):
            dy = cy + 4 + offset + j * 13
            parts.append(
                f'<text x="{cx}" y="{dy}" text-anchor="middle" '
                f'fill="#E4E4E7" font-size="11.5" font-weight="600">{line}</text>'
            )

        parts.append(
            f'<text x="{cx}" y="{_TOP + _NODE_H + 20}" text-anchor="middle" '
            f'fill="#7C7C87" font-size="9.5">{caption}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def render_pipeline_diagram() -> None:
    st.caption(
        "Fluxo completo: do PDF indexado à resposta com citação de fonte. "
        "Toda a inferência acontece localmente."
    )
    st.markdown(f'<div class="pipeline-wrap">{_build_svg()}</div>', unsafe_allow_html=True)
