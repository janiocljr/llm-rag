from __future__ import annotations

import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from components.chat import (
    ASSISTANT_AVATAR,
    render_chat_history,
    render_empty_state,
    render_suggestions,
    render_user_message,
)
from components.diagram import render_pipeline_diagram
from components.prompt_viewer import render_prompt_viewer
from components.sidebar import render_sidebar
from components.thinking import render_sources_panel
from utils.api_client import APIClient
from utils.demo_data import DEMO_QUERY_RESPONSE
from utils.formatting import format_latency


st.set_page_config(
    page_title="RAG Chat — Q&A de documentos",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = (Path(__file__).parent / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None


def _normalize_prompt(raw) -> str:
    if isinstance(raw, list):
        return "\n\n".join(
            f"[{m.get('role', '').upper()}]\n{m.get('content', '')}" for m in raw
        )
    return raw or ""


def _append_assistant(content: str, meta: dict) -> None:
    st.session_state.messages.append(
        {"role": "assistant", "content": content, "meta": meta}
    )


def _answer_demo(question: str) -> None:
    response = DEMO_QUERY_RESPONSE
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        placeholder = st.empty()
        buf = ""
        for word in response["answer"].split(" "):
            buf += word + " "
            placeholder.markdown(buf + "▌")
            time.sleep(0.012)
        placeholder.markdown(response["answer"])

    st.session_state.last_response = response
    _append_assistant(
        response["answer"],
        {
            "latency_ms": response.get("latency_ms", 0),
            "found_in_documents": response.get("found_in_documents", False),
            "chunks": response.get("retrieved_chunks", []),
        },
    )
    st.rerun()


def _answer_live(client: APIClient, question: str, top_k: int, threshold: float) -> None:
    t0 = time.perf_counter()
    buf = ""
    chunks: list[dict] = []
    full_prompt = ""
    error: str | None = None

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        placeholder = st.empty()
        placeholder.markdown("*Buscando trechos relevantes…*")

        for event in client.query_stream(
            question=question,
            top_k=top_k,
            similarity_threshold=threshold,
        ):
            etype = event.get("type")
            if etype == "meta":
                chunks = event.get("retrieved_chunks", [])
                full_prompt = _normalize_prompt(event.get("full_prompt", ""))
                if not buf:
                    placeholder.markdown(
                        f"*{len(chunks)} trecho(s) recuperado(s) — gerando resposta…*"
                    )
            elif etype == "token":
                buf += event.get("text", "")
                placeholder.markdown(buf + "▌")
            elif etype in ("complete", "done"):
                buf += event.get("text", "")
                break
            elif etype == "error":
                error = event.get("error", "Erro desconhecido ao consultar a API.")
                break

    if not buf and not error:
        error = "A API encerrou a conexão sem retornar resposta. Tente novamente."

    latency_ms = (time.perf_counter() - t0) * 1000

    if error:
        _append_assistant(error, {"error": True})
    else:
        found = bool(chunks)
        st.session_state.last_response = {
            "answer": buf,
            "latency_ms": latency_ms,
            "found_in_documents": found,
            "retrieved_chunks": chunks,
            "full_prompt": full_prompt,
        }
        _append_assistant(
            buf,
            {"latency_ms": latency_ms, "found_in_documents": found, "chunks": chunks},
        )
    st.rerun()


sidebar_state = render_sidebar()
client = APIClient(base_url=sidebar_state.api_url)

typed = st.chat_input("Pergunte algo sobre os documentos indexados…")
question = typed or st.session_state.pop("pending_question", None)

if not st.session_state.messages and not question:
    render_empty_state()
    suggestion = render_suggestions()
    if suggestion:
        st.session_state.pending_question = suggestion
        st.rerun()
    if sidebar_state.show_diagram:
        st.markdown("")
        with st.expander("Como funciona o pipeline RAG"):
            render_pipeline_diagram()
else:
    render_chat_history(st.session_state.messages)

if question and question.strip():
    q = question.strip()
    st.session_state.messages.append({"role": "user", "content": q, "meta": {}})
    render_user_message(q)
    if sidebar_state.demo_mode:
        _answer_demo(q)
    else:
        _answer_live(client, q, sidebar_state.top_k, sidebar_state.similarity_threshold)

last = st.session_state.last_response
if last is not None and st.session_state.messages:
    st.divider()
    st.markdown('<p class="panel-title">Detalhes da última consulta</p>', unsafe_allow_html=True)

    latency = last.get("latency_ms", 0)
    if latency:
        st.caption(f"Latência total: {format_latency(latency)}")

    chunks = last.get("retrieved_chunks", [])
    has_prompt = bool(sidebar_state.show_prompt and last.get("full_prompt"))

    labels = [f"Fontes ({len(chunks)})"]
    if has_prompt:
        labels.append("Prompt do LLM")
    if sidebar_state.show_diagram:
        labels.append("Pipeline")

    tabs = st.tabs(labels)
    with tabs[0]:
        render_sources_panel(chunks)

    next_tab = 1
    if has_prompt:
        with tabs[next_tab]:
            render_prompt_viewer(last["full_prompt"])
        next_tab += 1
    if sidebar_state.show_diagram:
        with tabs[next_tab]:
            render_pipeline_diagram()
