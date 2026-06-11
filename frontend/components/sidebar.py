from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from utils.api_client import APIClient
from utils.demo_data import DEMO_STATS
from utils.formatting import format_latency


@dataclass
class SidebarState:
    api_url: str = "http://localhost:8000"
    top_k: int = 10
    similarity_threshold: float = 0.45
    demo_mode: bool = False
    show_prompt: bool = True
    show_diagram: bool = True


_BRAND_ICON = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="white">'
    '<path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z"/>'
    "</svg>"
)


def _section(title: str) -> None:
    st.markdown(
        f'<p class="sidebar-section-title">{title}</p>', unsafe_allow_html=True
    )


def _render_brand() -> None:
    st.markdown(
        '<div class="sidebar-brand">'
        f'<div class="brand-mark" aria-hidden="true">{_BRAND_ICON}</div>'
        "<div>"
        '<div class="brand-name">RAG Chat</div>'
        '<div class="brand-tag">Q&amp;A offline de documentos</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_connection_status(demo_mode: bool) -> None:
    if demo_mode:
        st.markdown(
            '<div class="status-row"><span class="status-dot ok"></span>'
            "Modo demo ativo — sem API</div>",
            unsafe_allow_html=True,
        )
        return

    conn = st.session_state.get("conn_status")
    if conn is None:
        st.markdown(
            '<div class="status-row"><span class="status-dot unknown"></span>'
            "Conexão não verificada</div>",
            unsafe_allow_html=True,
        )
        return

    dot = "ok" if conn["ok"] else "err"
    st.markdown(
        f'<div class="status-row"><span class="status-dot {dot}"></span>'
        f'{conn["label"]}</div>',
        unsafe_allow_html=True,
    )
    if conn.get("detail"):
        st.caption(conn["detail"])


def render_sidebar() -> SidebarState:
    state = SidebarState()

    with st.sidebar:
        _render_brand()

        if st.button("Nova conversa", icon=":material/add:", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_response = None
            st.rerun()

        _section("Conexão")
        state.api_url = st.text_input("URL da API", value="http://localhost:8000")
        state.demo_mode = st.toggle(
            "Modo demo (offline)",
            value=False,
            help="Responde com dados de exemplo, sem precisar da API.",
        )

        if not state.demo_mode and st.button(
            "Testar conexão", icon=":material/network_check:", use_container_width=True
        ):
            client = APIClient(base_url=state.api_url)
            health = client.health()
            if "error" in health:
                st.session_state.conn_status = {
                    "ok": False,
                    "label": "API indisponível",
                    "detail": health["error"],
                }
            else:
                st.session_state.conn_status = {
                    "ok": True,
                    "label": f"API online · v{health.get('version', '?')}",
                    "detail": "",
                }
        _render_connection_status(state.demo_mode)

        _section("Recuperação")
        state.top_k = st.slider(
            "Top-K chunks candidatos",
            min_value=5, max_value=30, value=20,
            help="Quantos candidatos buscar antes do re-ranking MMR. "
                 "Aumentar para mais cobertura, diminuir para mais velocidade."
        )
        state.similarity_threshold = st.slider(
            "Limiar de similaridade",
            min_value=0.20, max_value=0.90, value=0.45, step=0.05,
            help="Score mínimo coseno para um chunk virar candidato. "
                 "Com e5-small e prefixos corretos, os scores ficam tipicamente entre "
                 "0.70 e 0.90 — valores até ~0.70 quase não filtram (mais cobertura); "
                 "use 0.80+ para filtrar agressivamente em buscas pontuais.",
        )

        _section("Índice FAISS")
        if st.button(
            "Atualizar estatísticas", icon=":material/monitoring:", use_container_width=True
        ):
            stats = (
                DEMO_STATS
                if state.demo_mode
                else APIClient(base_url=state.api_url).stats()
            )
            if "error" in stats:
                st.error(stats["error"])
            else:
                st.session_state.index_stats = stats

        stats = st.session_state.get("index_stats")
        if stats:
            col1, col2 = st.columns(2)
            col1.metric("Chunks", stats.get("total_chunks", 0))
            col2.metric("Documentos", len(stats.get("documents", [])))
            docs = stats.get("documents", [])
            if docs:
                with st.expander("Documentos indexados"):
                    for doc in docs:
                        st.caption(f"• {doc}")

        if st.button("Re-ingerir PDFs", icon=":material/refresh:", use_container_width=True):
            if state.demo_mode:
                st.toast("Re-ingestão desabilitada no modo demo.", icon="ℹ️")
            else:
                client = APIClient(base_url=state.api_url)
                with st.spinner("Ingerindo PDFs…"):
                    result = client.ingest(force_reindex=True)
                if "error" in result:
                    st.error(result["error"])
                else:
                    summary = (
                        f"{result['chunks_indexed']} chunks · "
                        f"{format_latency(result['latency_ms'])}"
                    )
                    st.session_state.last_ingest = summary
                    st.toast(f"Ingestão concluída: {summary}", icon="✅")

        if st.session_state.get("last_ingest"):
            st.caption(f"Última ingestão: {st.session_state.last_ingest}")

        _section("Exibição")
        state.show_prompt = st.toggle("Mostrar prompt completo do LLM", value=True)
        state.show_diagram = st.toggle("Mostrar diagrama do pipeline", value=True)

        st.divider()
        st.caption("RAG System · v1.0.0 · Inferência 100% local")

    return state
