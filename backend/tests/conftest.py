"""
tests/conftest.py
=================
Shared pytest fixtures and configuration for all tests.
Provides reusable test data, mocks, and settings.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

try:
    from app.core.ingestion import Chunk
except Exception:
    from app.models.schemas import DocumentChunk as Chunk

from app.core.config import settings as app_settings
from app.core.text_utils import estimate_tokens, clean_text


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
    """Generate normalized random vectors for testing."""
    rng = np.random.default_rng(42)
    dim = 384
    vecs = rng.random((len(sample_chunks), dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return """
    # Section 1: Introduction

    This is a paragraph with some text.
    It contains multiple lines and should be processed correctly.

    - Item 1
    - Item 2
    - Item 3

    ## Section 2: Details

    Another paragraph here with more content.
    Multiple sentences to test chunking algorithms.
    This text should be properly cleaned and processed.
    """


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder for testing."""
    rng = np.random.default_rng(42)
    mock = MagicMock()
    mock.embed_query = MagicMock(
        return_value=rng.standard_normal(1024).astype(np.float32)
    )
    mock.embed_documents = MagicMock(
        return_value=rng.standard_normal((3, 1024)).astype(np.float32)
    )
    return mock


@pytest.fixture
def mock_llm():
    """Provide a mock LLM for testing."""
    mock = MagicMock()
    mock.generate = MagicMock(
        return_value="This is a test response from the LLM."
    )
    return mock


@pytest.fixture
def temp_pdf_path(tmp_path):
    """Provide a temporary PDF file path."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%Test PDF Content")
    return pdf_path


@pytest.fixture
def temp_data_dir(tmp_path):
    """Provide a temporary data directory with proper structure."""
    data_dir = tmp_path / "data"
    (data_dir / "pdfs").mkdir(parents=True)
    (data_dir / "index").mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_text_with_headers() -> str:
    """Provide sample text with headers and footers."""
    return """
    === CONFIDENTIAL REPORT - PAGE 1 ===

    # Main Section

    This is the main content of the page.
    Important information goes here.

    === PAGE 1 | Company Inc. ===
    """
