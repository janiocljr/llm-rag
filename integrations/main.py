"""
app/main.py  —  v3 (ChromaDB + MongoDB)
========================================
FastAPI application factory.

Lifespan
--------
1. Initialise RAGPipeline (loads embedder + LLM + connects to ChromaDB/MongoDB).
2. Attach pipeline and memory orchestrator to app.state.
3. Auto-ingest PDFs on first boot if the vector store is empty.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as rag_router
from app.api.memory_routes import router as memory_router
from app.core.config import settings
from app.core.pipeline import RAGPipeline
from app.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG system v3 (ChromaDB + MongoDB)")

    pipeline = RAGPipeline(settings)
    app.state.pipeline = pipeline
    app.state.memory   = pipeline.memory

    if pipeline.vector_store.size == 0:
        logger.info("Vector store empty — running auto-ingest…")
        try:
            result = pipeline.ingest(force_reindex=False)
            logger.info(
                f"Auto-ingest: {result.chunks_indexed} chunks from "
                f"{result.documents_processed} documents"
            )
        except FileNotFoundError:
            logger.warning("No PDFs found in data_dir — skipping auto-ingest")
        except Exception as exc:
            logger.error(f"Auto-ingest failed: {exc}", exc_info=True)
    else:
        logger.info(f"Vector store loaded: {pipeline.vector_store.size} chunks")

    yield

    logger.info("RAG system shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Chat LLM — Persistent Memory",
        version="3.0.0",
        description=(
            "Retrieval-Augmented Generation with persistent memory.\n\n"
            "**Vector store:** ChromaDB (semantic search)\n"
            "**Document store:** MongoDB (knowledge base, sessions, tasks)\n"
            "**LLM:** Mistral 7B Instruct (local, offline)"
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(rag_router, prefix="/api/v1")

    app.include_router(memory_router, prefix="/api/v1")

    @app.get("/health", tags=["system"])
    async def health():
        return {"status": "ok", "version": "3.0.0"}

    return app


app = create_app()
