from __future__ import annotations

import streamlit as st

from utils.formatting import score_color, truncate


def _chunk_card(chunk: dict, index: int = 1) -> str:
    score = chunk.get("score", 0.0)
    source = chunk.get("source_file", "?")
    page = chunk.get("page_number", "?")
    text = chunk.get("text", "")
    text_preview = truncate(text, max_words=80)
    citation = chunk.get("citation", f"[{source}, p. {page}]")

    relevance_pct = int(score * 100)

    if score > 0.6:
        color = "4CAF50"
    elif score > 0.5:
        color = "FFC107"
    else:
        color = "FF6B6B"

    return (
        f'<div class="chunk-card" style="border-left: 4px solid #{color}">'
        f'  <div class="chunk-header">'
        f'    <span class="chunk-source">📄 Fonte {index}: {citation}</span>'
        f'    <span class="score-badge" style="background: #{color}; color: white">'
        f'      {relevance_pct}%'
        f'    </span>'
        f'  </div>'
        f'  <div class="chunk-text">"{text_preview}"</div>'
        f'  <div style="font-size: 0.8em; color: #666; margin-top: 8px;">'
        f'    📊 Similaridade: {score:.1%}'
        f'  </div>'
        f"</div>"
    )


def render_thinking_panel(chunks: list[dict]) -> None:
    if not chunks:
        st.info("Nenhum chunk recuperado nesta consulta.")
        return

    label = f"🧠 Raciocínio — {len(chunks)} chunk(s) utilizado(s)"
    with st.expander(label, expanded=True):
        st.markdown(
            "Trechos dos documentos usados como contexto pelo LLM:",
            help="Ordenados pelo score MMR (relevância + diversidade).",
        )
        cards_html = "".join(_chunk_card(c, i) for i, c in enumerate(chunks, 1))
        st.markdown(cards_html, unsafe_allow_html=True)
