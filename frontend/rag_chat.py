from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from components.chat import render_chat_history, render_empty_state
from components.diagram import render_pipeline_diagram
from components.prompt_viewer import render_prompt_viewer
from components.sidebar import render_sidebar
from components.thinking import render_thinking_panel
from utils.api_client import APIClient
from utils.demo_data import DEMO_QUERY_RESPONSE, DEMO_STATS
from utils.formatting import format_latency


st.set_page_config(
    page_title="RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


_CSS = (Path(__file__).parent / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False


sidebar_state = render_sidebar()

client = APIClient(base_url=sidebar_state.api_url)
demo_mode = sidebar_state.demo_mode
_, col_main, _ = st.columns([1, 3, 1])

with col_main:
    st.markdown("## 💬 RAG Chat")

    if not st.session_state.messages:
        render_empty_state()
    else:
        render_chat_history(st.session_state.messages)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Sua pergunta",
            placeholder="Ex.: Quais são os cenários de crescimento do PIB do Paraná?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Enviar ➤", use_container_width=True)

    if submitted and user_input.strip():
        question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": question, "meta": {}})

        with st.spinner("Consultando documentos…"):
            if demo_mode:
                response = DEMO_QUERY_RESPONSE
                st.session_state.last_response = response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "meta": {
                        "latency_ms": response.get("latency_ms", 0),
                        "found_in_documents": response.get("found_in_documents", False),
                        "chunks": response.get("retrieved_chunks", []),
                    },
                })
                st.rerun()


            placeholder = st.empty()
            assistant_meta = {"latency_ms": 0, "found_in_documents": True, "chunks": []}
            content_buf = ""

            for event in client.query_stream(
                question=question,
                top_k=sidebar_state.top_k,
                similarity_threshold=sidebar_state.similarity_threshold,
            ):
                if event.get("type") == "meta":
                    assistant_meta["chunks"] = event.get("retrieved_chunks", [])

                    placeholder.markdown(
                        f'<div class="bubble-ai">{content_buf}</div>', unsafe_allow_html=True
                    )
                elif event.get("type") == "token":
                    content_buf += event.get("text", "")
                    placeholder.markdown(
                        f'<div class="bubble-ai">{content_buf}</div>', unsafe_allow_html=True
                    )
                elif event.get("type") in ("complete", "done"):

                    content_buf += event.get("text", "")
                    st.session_state.last_response = {
                        "answer": content_buf,
                        "latency_ms": 0,
                        "found_in_documents": True,
                        "retrieved_chunks": assistant_meta.get("chunks", []),
                        "full_prompt": event.get("full_prompt", ""),
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": content_buf,
                        "meta": {
                            "latency_ms": 0,
                            "found_in_documents": True,
                            "chunks": assistant_meta.get("chunks", []),
                        },
                    })
                    placeholder.empty()
                    st.rerun()
                elif event.get("type") == "error":
                    placeholder.markdown(
                        f'<div class="bubble-ai not-found">Error: {event.get("error")}</div>',
                        unsafe_allow_html=True,
                    )
                    break

    last = st.session_state.last_response
    if last is not None:
        st.divider()
        st.markdown("## 🔍 Detalhes da Consulta")
        latency_str = format_latency(last.get("latency_ms", 0))
        st.caption(f"⏱ Latência total: **{latency_str}**")

        render_thinking_panel(last.get("retrieved_chunks", []))

        if sidebar_state.show_prompt and last.get("full_prompt"):
            render_prompt_viewer(last["full_prompt"])

    if sidebar_state.show_diagram:
        st.divider()
        render_pipeline_diagram()
