from __future__ import annotations

import streamlit as st


def render_prompt_viewer(full_prompt: str) -> None:
    st.caption(
        "Prompt exato construído e enviado ao modelo de linguagem. "
        "Use o ícone no canto do bloco para copiar."
    )
    st.code(full_prompt, language=None, wrap_lines=True, height=360)

    word_count = len(full_prompt.split())
    st.caption(f"{word_count} palavras · {len(full_prompt)} caracteres")
