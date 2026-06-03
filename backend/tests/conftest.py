"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest

try:

    from app.core.ingestion import Chunk  #type: ignore
except Exception:
    from app.models.schemas import DocumentChunk as Chunk  #type: ignore
from app.core.config import settings as app_settings


@pytest.fixture
def settings():
    return app_settings


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc_a_p1_c0",
            source_file="doc_a.pdf",
            page_number=1,
            chunk_index=0,
            text="O PIB do Paraná cresceu 2% no cenário tendencial.",
        ),
        Chunk(
            chunk_id="doc_a_p1_c1",
            source_file="doc_a.pdf",
            page_number=1,
            chunk_index=1,
            text="A taxa de inadimplência ficou em 3,5% em junho de 2025.",
        ),
        Chunk(
            chunk_id="doc_b_p2_c0",
            source_file="doc_b.pdf",
            page_number=2,
            chunk_index=0,
            text="Os investimentos em infraestrutura aumentaram 15% no último trimestre.",
        ),
    ]


@pytest.fixture
def sample_vectors(sample_chunks) -> np.ndarray:
    rng = np.random.default_rng(42)
    dim = 384
    vecs = rng.random((len(sample_chunks), dim)).astype(np.float32)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms
