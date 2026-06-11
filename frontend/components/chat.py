from __future__ import annotations

import streamlit as st

from utils.formatting import format_latency

ASSISTANT_AVATAR = ":material/auto_awesome:"

SUGGESTED_QUESTIONS = [
    "Quais são os cenários de crescimento do PIB do Paraná?",
    "Quais são os principais setores econômicos do Paraná?",
    "Qual foi a inflação acumulada em 12 meses em julho de 2025?",
    "Qual é o objetivo principal da avaliação de políticas públicas?",
]

_NOT_FOUND_BANNER = (
    '<div class="notice-warn" role="status">'
    '<span aria-hidden="true">⚠</span>'
    "<span>Resposta gerada sem suporte nos documentos indexados — confira as fontes.</span>"
    "</div>"
)

_HERO_ICON = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
    'stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>'
    "</svg>"
)


def render_user_message(text: str) -> None:
    with st.chat_message("user"):
        st.markdown(text)


def render_assistant_message(text: str, meta: dict) -> None:
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if meta.get("error"):
            st.markdown(
                f'<div class="notice-error" role="alert">'
                f'<span aria-hidden="true">✕</span><span>{text}</span></div>',
                unsafe_allow_html=True,
            )
            return

        if not meta.get("found_in_documents", True):
            st.markdown(_NOT_FOUND_BANNER, unsafe_allow_html=True)

        st.markdown(text)

        meta_parts: list[str] = []
        latency_ms = meta.get("latency_ms", 0)
        if latency_ms:
            meta_parts.append(f"⏱ {format_latency(latency_ms)}")
        n_chunks = len(meta.get("chunks", []))
        if n_chunks:
            meta_parts.append(f"{n_chunks} fonte(s) consultada(s)")
        if meta_parts:
            st.markdown(
                f'<div class="msg-meta">{" · ".join(meta_parts)}</div>',
                unsafe_allow_html=True,
            )


def render_chat_history(messages: list[dict]) -> None:
    for msg in messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_assistant_message(msg["content"], msg.get("meta", {}))


def render_empty_state() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-mark" aria-hidden="true">{_HERO_ICON}</div>
            <div class="hero-badge">
                <span aria-hidden="true">●</span> 100% offline · inferência local
            </div>
            <h1>Pergunte aos seus documentos</h1>
            <p>O sistema recupera os trechos mais relevantes dos PDFs indexados
            e gera uma resposta com citação de fonte.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggestions() -> str | None:
    st.markdown('<p class="hero-hint">Experimente perguntar</p>', unsafe_allow_html=True)

    clicked: str | None = None
    with st.container(key="suggestions"):
        cols = st.columns(2)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                if st.button(question, key=f"suggestion_{i}", use_container_width=True):
                    clicked = question
    return clicked
