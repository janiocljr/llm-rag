"""
app/models/schemas.py
=====================
Pydantic models for API request/response validation and
internal data structures used across the pipeline.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field






class DocumentChunk(BaseModel):
    """
    A single piece of text extracted from a PDF.

    METADATA RATIONALE
    ------------------
    Tracking source, page and chunk index allows us to:
    1. Cite the exact location of every piece of evidence in the answer.
    2. Debug retrieval quality (which pages are most often retrieved?).
    3. De-duplicate chunks from the same page in MMR re-ranking.
    """

    chunk_id: str = Field(description="Globally unique identifier: '<doc_id>_p<page>_c<idx>'")
    text: str = Field(description="Cleaned chunk text.")
    source_file: str = Field(description="Original PDF filename (e.g. 'report_2024.pdf').")
    page_number: int = Field(description="1-based page number within the PDF.")
    chunk_index: int = Field(description="0-based index of this chunk within its page.")
    total_chunks_in_page: int = Field(description="Total chunks on this page (for position context).")
    char_count: int = Field(description="Number of characters in the chunk.")
    token_estimate: int = Field(description="Rough token count (chars / 4).")

    is_table: bool = Field(default=False, description="True for chunks generated from detected tables.")
    table_csv_path: Optional[str] = Field(default=None, description="Filesystem path to CSV extracted from table, if any.")

    semantic_type: Optional[str] = Field(default=None, description="Semantic classification: heading, paragraph, list, table, etc.")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords from chunk content.")
    semantic_group_id: Optional[str] = Field(default=None, description="ID of semantic pseudo-document group this chunk belongs to.")

    @property
    def citation(self) -> str:
        """Human-readable citation string, e.g. '[report_2024.pdf, p. 3]'."""
        return f"[{self.source_file}, p. {self.page_number}]"


class RetrievedChunk(BaseModel):
    """A DocumentChunk augmented with its retrieval similarity score."""

    chunk: DocumentChunk
    score: float = Field(description="Cosine similarity score in [0, 1].")

    @property
    def formatted(self) -> str:
        """Ready-to-inject text block for prompt construction."""
        return (
            f"--- Source: {self.chunk.source_file} | Page: {self.chunk.page_number} "
            f"| Score: {self.score:.3f} ---\n{self.chunk.text}"
        )






class QueryRequest(BaseModel):
    """Incoming user query."""

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="The user's natural-language question.",
        examples=["What are the main conclusions of the annual report?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the default number of chunks to retrieve.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the default similarity threshold.",
    )


class RetrievedChunkResponse(BaseModel):
    """Serialisable representation of a retrieved chunk for the API response."""

    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    score: float
    text: str
    citation: str


class QueryResponse(BaseModel):
    """
    Full RAG pipeline output returned to the client.

    Returning the full prompt and retrieved chunks gives operators
    full transparency into what the model saw — essential for debugging
    hallucinations and evaluating retrieval quality.
    """

    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]
    full_prompt: str
    found_in_documents: bool = Field(
        description="False when similarity scores are all below the threshold."
    )
    latency_ms: float


class IngestRequest(BaseModel):
    """Trigger document ingestion (useful for CI / initial setup)."""

    force_reindex: bool = Field(
        default=False,
        description="If True, delete the existing index and rebuild from scratch.",
    )


class IngestResponse(BaseModel):
    chunks_indexed: int
    documents_processed: int
    latency_ms: float


class IndexStatsResponse(BaseModel):
    total_chunks: int
    documents: list[str]
    index_type: str
    embedding_model: str
    embedding_dim: int
