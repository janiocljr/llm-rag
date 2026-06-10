from __future__ import annotations

import streamlit as st


def _bubble_user(text: str) -> None:
    st.markdown(
        f'<div class="bubble-user">{text}</div>',
        unsafe_allow_html=True,
    )


def _bubble_ai(text: str, meta: dict) -> None:
    found = meta.get("found_in_documents", True)
    latency_ms = meta.get("latency_ms", 0)
    extra_class = "" if found else " not-found"

    not_found_banner = ""
    if not found:
        not_found_banner = (
            '<div style="font-size:0.8rem;color:#e74c3c;margin-bottom:6px;">'
            "⚠️ Informação não encontrada nos documentos indexados."
            "</div>"
        )

    meta_row = (
        f'<div class="bubble-meta">⏱ {latency_ms:.0f} ms</div>'
        if latency_ms
        else ""
    )

    st.markdown(
        f'<div class="bubble-ai{extra_class}">'
        f"{not_found_banner}"
        f"{text}"
        f"{meta_row}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_chat_history(messages: list[dict]) -> None:
    for msg in messages:
        if msg["role"] == "user":
            _bubble_user(msg["content"])
        else:
            _bubble_ai(msg["content"], msg.get("meta", {}))


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size:3rem;">🤖</div>
            <h3>RAG Chat — Offline PDF Q&amp;A</h3>
            <p>Faça uma pergunta sobre os documentos indexados.<br>
            O sistema irá recuperar os trechos mais relevantes e gerar uma resposta.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
