"""
RAG System - Main FastAPI Application
======================================
Entry point for the offline RAG system.
All components run locally — zero external API calls.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api import memory_routes
from app.core.config import settings
from app.core.pipeline import RAGPipeline
from app.core.memory import MemoryOrchestrator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — warm up pipeline on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    - Instantiates the RAG pipeline at startup (loads models into memory).
    - Exposes it via app.state so routes can access it.
    """
    logger.info("🚀  Starting RAG system — loading models...")
    t0 = time.perf_counter()

    pipeline = RAGPipeline(settings)
    app.state.pipeline = pipeline

    # Memory orchestrator (ChromaDB + MongoDB) — optional integration
    try:
        mem = MemoryOrchestrator(pipeline.embedder)
        app.state.memory = mem
        # mount memory routes
        app.include_router(memory_routes.router, prefix="/api/v1")
        logger.info("✅ Memory orchestrator mounted: /api/v1/memory")
    except Exception:
        logger.info("Memory orchestrator not available or failed to initialize")

    elapsed = time.perf_counter() - t0
    logger.info(f"✅  Pipeline ready in {elapsed:.1f}s")

    yield  # ← application runs here

    logger.info("🛑  Shutting down RAG system")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG System — Offline PDF Q&A",
    description=(
        "Retrieval-Augmented Generation over local PDF documents. "
        "100 %% offline: no external APIs, no internet required."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health():
    """Quick liveness probe — no heavy operations."""
    return {"status": "ok", "version": "1.0.0"}
