"""Prompt viewer — expandable raw LLM prompt panel."""
from __future__ import annotations

import streamlit as st


def render_prompt_viewer(full_prompt: str) -> None:
    """Render the raw LLM prompt inside a collapsible expander."""
    with st.expander("📝 Prompt completo enviado ao LLM", expanded=False):
        st.markdown(
            "Prompt exato que foi construído e enviado ao modelo de linguagem:",
        )

        escaped = (
            full_prompt
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        st.markdown(
            f'<div class="prompt-box">{escaped}</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            word_count = len(full_prompt.split())
            st.caption(f"📏 {word_count} palavras · {len(full_prompt)} chars")
        with col2:
            if st.button("📋 Copiar prompt", use_container_width=True):
                st.write(
                    f"<script>navigator.clipboard.writeText(`{full_prompt}`)</script>",
                    unsafe_allow_html=True,
                )
