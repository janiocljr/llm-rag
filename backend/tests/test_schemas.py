"""
Unit tests for app.models.schemas module.

Tests cover:
- DocumentChunk creation and properties
- RetrievedChunk creation and properties
- Query/Response models
- Ingest models
- Validation
"""

import pytest

from app.models.schemas import (
    DocumentChunk,
    RetrievedChunk,
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    IndexStatsResponse,
    RetrievedChunkResponse,
)


class TestDocumentChunk:
    """Test DocumentChunk model."""

    def test_document_chunk_creation(self) -> None:
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            chunk_id="doc_1_p1_c0",
            text="Sample text content",
            source_file="report.pdf",
            page_number=1,
            chunk_index=0,
            total_chunks_in_page=3,
            char_count=19,
            token_estimate=5,
        )
        assert chunk.chunk_id == "doc_1_p1_c0"
        assert chunk.text == "Sample text content"
        assert chunk.source_file == "report.pdf"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0
        assert chunk.total_chunks_in_page == 3
        assert chunk.char_count == 19
        assert chunk.token_estimate == 5

    def test_document_chunk_citation_property(self) -> None:
        """Test citation property of DocumentChunk."""
        chunk = DocumentChunk(
            chunk_id="doc_1_p5_c0",
            text="Test",
            source_file="annual_report_2024.pdf",
            page_number=5,
            chunk_index=0,
            total_chunks_in_page=1,
            char_count=4,
            token_estimate=1,
        )
        assert chunk.citation == "[annual_report_2024.pdf, p. 5]"

    def test_document_chunk_with_table_metadata(self) -> None:
        """Test DocumentChunk with table metadata."""
        chunk = DocumentChunk(
            chunk_id="doc_1_p2_c0",
            text="| Header | Value |\n|--------|-------|\n| A | 1 |",
            source_file="data.pdf",
            page_number=2,
            chunk_index=0,
            total_chunks_in_page=1,
            char_count=40,
            token_estimate=10,
            is_table=True,
            table_csv_path="/data/index/tables/table_001.csv",
        )
        assert chunk.is_table is True
        assert chunk.table_csv_path == "/data/index/tables/table_001.csv"

    def test_document_chunk_with_semantic_type(self) -> None:
        """Test DocumentChunk with semantic type."""
        chunk = DocumentChunk(
            chunk_id="doc_1_p1_c0",
            text="Introduction to the report",
            source_file="report.pdf",
            page_number=1,
            chunk_index=0,
            total_chunks_in_page=5,
            char_count=26,
            token_estimate=6,
            semantic_type="heading",
            keywords=["introduction", "report"],
        )
        assert chunk.semantic_type == "heading"
        assert chunk.keywords == ["introduction", "report"]

    def test_document_chunk_default_optional_fields(self) -> None:
        """Test DocumentChunk default optional fields."""
        chunk = DocumentChunk(
            chunk_id="test",
            text="text",
            source_file="file.pdf",
            page_number=1,
            chunk_index=0,
            total_chunks_in_page=1,
            char_count=4,
            token_estimate=1,
        )
        assert chunk.is_table is False
        assert chunk.table_csv_path is None
        assert chunk.semantic_type is None
        assert chunk.keywords == []
        assert chunk.semantic_group_id is None


class TestRetrievedChunk:
    """Test RetrievedChunk model."""

    def test_retrieved_chunk_creation(self) -> None:
        """Test creating a RetrievedChunk."""
        chunk = DocumentChunk(
            chunk_id="doc_1_p1_c0",
            text="Content here",
            source_file="doc.pdf",
            page_number=1,
            chunk_index=0,
            total_chunks_in_page=1,
            char_count=13,
            token_estimate=3,
        )
        retrieved = RetrievedChunk(chunk=chunk, score=0.85)
        assert retrieved.chunk == chunk
        assert retrieved.score == 0.85

    def test_retrieved_chunk_formatted_property(self) -> None:
        """Test formatted property of RetrievedChunk."""
        chunk = DocumentChunk(
            chunk_id="doc_1_p3_c0",
            text="Some important content",
            source_file="report.pdf",
            page_number=3,
            chunk_index=0,
            total_chunks_in_page=1,
            char_count=22,
            token_estimate=5,
        )
        retrieved = RetrievedChunk(chunk=chunk, score=0.92)
        formatted = retrieved.formatted
        assert "report.pdf" in formatted
        assert "Page: 3" in formatted
        assert "0.920" in formatted
        assert "Some important content" in formatted

    def test_retrieved_chunk_score_validation(self) -> None:
        """Test score validation for RetrievedChunk."""
        chunk = DocumentChunk(
            chunk_id="test",
            text="text",
            source_file="file.pdf",
            page_number=1,
            chunk_index=0,
            total_chunks_in_page=1,
            char_count=4,
            token_estimate=1,
        )
        retrieved = RetrievedChunk(chunk=chunk, score=0.5)
        assert 0 <= retrieved.score <= 1


class TestQueryRequest:
    """Test QueryRequest model."""

    def test_query_request_minimal(self) -> None:
        """Test creating a minimal QueryRequest."""
        req = QueryRequest(question="What is the main topic?")
        assert req.question == "What is the main topic?"
        assert req.top_k is None
        assert req.similarity_threshold is None

    def test_query_request_with_overrides(self) -> None:
        """Test QueryRequest with parameter overrides."""
        req = QueryRequest(
            question="What happened?",
            top_k=10,
            similarity_threshold=0.5,
        )
        assert req.question == "What happened?"
        assert req.top_k == 10
        assert req.similarity_threshold == 0.5

    def test_query_request_question_validation_min_length(self) -> None:
        """Test question minimum length validation."""
        with pytest.raises(ValueError):
            QueryRequest(question="ab")  # Too short

    def test_query_request_question_validation_max_length(self) -> None:
        """Test question maximum length validation."""
        long_question = "a" * 2001
        with pytest.raises(ValueError):
            QueryRequest(question=long_question)

    def test_query_request_top_k_validation(self) -> None:
        """Test top_k validation."""
        req_min = QueryRequest(question="test question", top_k=1)
        assert req_min.top_k == 1

        req_max = QueryRequest(question="test question", top_k=20)
        assert req_max.top_k == 20

    def test_query_request_similarity_threshold_validation(self) -> None:
        """Test similarity_threshold validation."""
        req_min = QueryRequest(question="test question", similarity_threshold=0.0)
        assert req_min.similarity_threshold == 0.0

        req_max = QueryRequest(question="test question", similarity_threshold=1.0)
        assert req_max.similarity_threshold == 1.0


class TestQueryResponse:
    """Test QueryResponse model."""

    def test_query_response_creation(self) -> None:
        """Test creating a QueryResponse."""
        resp = QueryResponse(
            question="Test question",
            answer="Test answer",
            retrieved_chunks=[],
            full_prompt="Full prompt",
            found_in_documents=True,
            latency_ms=150.5,
        )
        assert resp.question == "Test question"
        assert resp.answer == "Test answer"
        assert resp.retrieved_chunks == []
        assert resp.full_prompt == "Full prompt"
        assert resp.found_in_documents is True
        assert resp.latency_ms == 150.5

    def test_query_response_with_retrieved_chunks(self) -> None:
        """Test QueryResponse with retrieved chunks."""
        resp_chunk = RetrievedChunkResponse(
            chunk_id="c1",
            source_file="doc.pdf",
            page_number=1,
            chunk_index=0,
            score=0.9,
            text="Content",
            citation="[doc.pdf, p. 1]",
        )
        resp = QueryResponse(
            question="Q",
            answer="A",
            retrieved_chunks=[resp_chunk],
            full_prompt="Prompt",
            found_in_documents=True,
            latency_ms=100.0,
        )
        assert len(resp.retrieved_chunks) == 1
        assert resp.retrieved_chunks[0].chunk_id == "c1"


class TestIngestRequest:
    """Test IngestRequest model."""

    def test_ingest_request_default(self) -> None:
        """Test default IngestRequest."""
        req = IngestRequest()
        assert req.force_reindex is False

    def test_ingest_request_force_reindex(self) -> None:
        """Test IngestRequest with force_reindex=True."""
        req = IngestRequest(force_reindex=True)
        assert req.force_reindex is True


class TestIngestResponse:
    """Test IngestResponse model."""

    def test_ingest_response_creation(self) -> None:
        """Test creating an IngestResponse."""
        resp = IngestResponse(
            chunks_indexed=1000,
            documents_processed=5,
            latency_ms=5000.5,
        )
        assert resp.chunks_indexed == 1000
        assert resp.documents_processed == 5
        assert resp.latency_ms == 5000.5


class TestIndexStatsResponse:
    """Test IndexStatsResponse model."""

    def test_index_stats_response_creation(self) -> None:
        """Test creating an IndexStatsResponse."""
        resp = IndexStatsResponse(
            total_chunks=500,
            documents=["doc1.pdf", "doc2.pdf"],
            index_type="flat",
            embedding_model="BAAI/bge-m3",
            embedding_dim=1024,
        )
        assert resp.total_chunks == 500
        assert len(resp.documents) == 2
        assert resp.index_type == "flat"
        assert resp.embedding_model == "BAAI/bge-m3"
        assert resp.embedding_dim == 1024
