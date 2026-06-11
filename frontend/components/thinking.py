from __future__ import annotations

import html

import streamlit as st

from utils.formatting import score_tier, truncate


def _source_card(chunk: dict, index: int) -> str:
    score = chunk.get("score", 0.0)
    source = chunk.get("source_file", "?")
    page = chunk.get("page_number", "?")
    citation = chunk.get("citation", f"[{source}, p. {page}]")
    text_preview = truncate(chunk.get("text", ""), max_words=80)

    tier = score_tier(score)
    pct = max(0, min(100, int(round(score * 100))))
    citation = html.escape(citation)
    text_preview = html.escape(text_preview)

    return (
        f'<div class="source-card">'
        f'  <div class="source-head">'
        f'    <span class="source-title">'
        f'      <span class="source-index">#{index}</span>{citation}'
        f'    </span>'
        f'    <span class="score-pill {tier}">{pct}%</span>'
        f'  </div>'
        f'  <div class="relevance-track" role="img" aria-label="Similaridade de {pct}%">'
        f'    <div class="relevance-fill {tier}" style="width:{pct}%"></div>'
        f'  </div>'
        f'  <div class="source-text">“{text_preview}”</div>'
        f'</div>'
    )


def render_sources_panel(chunks: list[dict]) -> None:
    if not chunks:
        st.info(
            "Nenhum trecho foi recuperado nesta consulta. "
            "Tente reduzir o limiar de similaridade na barra lateral.",
            icon=":material/search_off:",
        )
        return

    st.caption(
        "Trechos usados como contexto pelo LLM, ordenados por score MMR "
        "(relevância + diversidade)."
    )
    cards_html = "".join(_source_card(c, i) for i, c in enumerate(chunks, 1))
    st.markdown(cards_html, unsafe_allow_html=True)
