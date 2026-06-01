"""Thinking panel — reasoning expander + retrieved chunk cards."""
from __future__ import annotations

import streamlit as st

from utils.formatting import score_color, truncate


def _chunk_card(chunk: dict) -> str:
    score = chunk.get("score", 0.0)
    color_class = score_color(score)
    source = chunk.get("source_file", "?")
    page = chunk.get("page_number", "?")
    text = truncate(chunk.get("text", ""), max_words=40)
    citation = chunk.get("citation", f"[{source}, p. {page}]")

    return (
        f'<div class="chunk-card">'
        f'  <div class="chunk-header">'
        f'    <span class="chunk-source">📄 {citation}</span>'
        f'    <span class="score-badge {color_class}">{score:.3f}</span>'
        f'  </div>'
        f'  <div class="chunk-text">"{text}"</div>'
        f"</div>"
    )


def render_thinking_panel(chunks: list[dict]) -> None:
    """Render the collapsible 'reasoning' panel with chunk cards."""
    if not chunks:
        st.info("Nenhum chunk recuperado nesta consulta.")
        return

    label = f"🧠 Raciocínio — {len(chunks)} chunk(s) utilizado(s)"
    with st.expander(label, expanded=True):
        st.markdown(
            "Trechos dos documentos usados como contexto pelo LLM:",
            help="Ordenados pelo score MMR (relevância + diversidade).",
        )
        cards_html = "".join(_chunk_card(c) for c in chunks)
        st.markdown(cards_html, unsafe_allow_html=True)
