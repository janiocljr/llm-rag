from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    total_chunks_in_page: int
    char_count: int
    token_estimate: int

    is_table: bool = Field(default=False)
    table_csv_path: Optional[str] = Field(default=None)

    semantic_type: Optional[str] = Field(default=None)
    keywords: list[str] = Field(default_factory=list)
    semantic_group_id: Optional[str] = Field(default=None)

    @property
    def citation(self) -> str:
        return f"[{self.source_file}, p. {self.page_number}]"


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float

    @property
    def formatted(self) -> str:
        return (
            f"--- Source: {self.chunk.source_file} | Page: {self.chunk.page_number} "
            f"| Score: {self.score:.3f} ---\n{self.chunk.text}"
        )


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    score: float
    text: str
    citation: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]
    full_prompt: str
    found_in_documents: bool
    latency_ms: float


class IngestRequest(BaseModel):
    force_reindex: bool = Field(default=False)


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
