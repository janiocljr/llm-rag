"""Sidebar component — returns a SidebarState dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from utils.api_client import APIClient
from utils.demo_data import DEMO_STATS
from utils.formatting import format_latency


@dataclass
class SidebarState:
    api_url: str = "http://localhost:8000"
    top_k: int = 12
    similarity_threshold: float = 0.45
    demo_mode: bool = False
    show_prompt: bool = True
    show_diagram: bool = True


def render_sidebar() -> SidebarState:
    """Render the full sidebar and return current state."""
    state = SidebarState()

    with st.sidebar:
        st.markdown("# 🤖 RAG Chat")
        st.markdown("---")

        # ── Connection ────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section-title">🔌 Conexão</p>', unsafe_allow_html=True)
        state.api_url = st.text_input("URL da API", value="http://localhost:8000")
        state.demo_mode = st.toggle("Modo demo (offline)", value=False)

        # Health check
        if not state.demo_mode:
            if st.button("🔍 Verificar conexão", use_container_width=True):
                client = APIClient(base_url=state.api_url)
                health = client.health()
                if "error" in health:
                    st.error(f"❌ {health['error']}")
                else:
                    st.success(f"✅ API online — v{health.get('version', '?')}")

        st.markdown("---")

        # ── Retrieval params ──────────────────────────────────────────
        st.markdown('<p class="sidebar-section-title">⚙️ Parâmetros de Recuperação</p>', unsafe_allow_html=True)
        state.top_k = st.slider("Top-K chunks candidatos", min_value=1, max_value=20, value=12)
        state.similarity_threshold = st.slider(
            "Limiar de similaridade",
            min_value=0.0, max_value=1.0, value=0.45, step=0.05,
            help="Score mínimo coseno para um chunk ser incluído na resposta.",
        )

        st.markdown("---")

        # ── Index management ──────────────────────────────────────────
        st.markdown('<p class="sidebar-section-title">📂 Índice FAISS</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Stats", use_container_width=True):
                client = APIClient(base_url=state.api_url)
                stats = DEMO_STATS if state.demo_mode else client.stats()
                if "error" in stats:
                    st.error(stats["error"])
                else:
                    st.metric("Chunks", stats.get("total_chunks", 0))
                    st.metric("Documentos", len(stats.get("documents", [])))
                    for doc in stats.get("documents", []):
                        st.caption(f"• {doc}")

        with col2:
            if st.button("🔄 Re-ingerir", use_container_width=True):
                if state.demo_mode:
                    st.info("Desabilitado no modo demo.")
                else:
                    client = APIClient(base_url=state.api_url)
                    with st.spinner("Ingerindo PDFs…"):
                        result = client.ingest(force_reindex=True)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(
                            f"✅ {result['chunks_indexed']} chunks em "
                            f"{format_latency(result['latency_ms'])}"
                        )

        st.markdown("---")

        # ── View options ──────────────────────────────────────────────
        st.markdown('<p class="sidebar-section-title">🖼 Visualização</p>', unsafe_allow_html=True)
        state.show_prompt = st.toggle("Mostrar prompt completo do LLM", value=True)
        state.show_diagram = st.toggle("Mostrar diagrama do pipeline", value=True)

        st.markdown("---")
        st.caption("RAG System · v1.0.0 · Offline PDF Q&A")

    return state
