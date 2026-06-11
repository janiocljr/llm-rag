import json
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def sample_text() -> str:
    return (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to natural intelligence displayed by animals including humans. "
        "AI research has been defined as the field of study of intelligent agents, "
        "which refers to any system that perceives its environment and takes actions "
        "that maximize its chance of achieving its goals.\n\n"
        "Machine learning (ML) is a subset of AI. ML algorithms build a model based "
        "on sample data, known as training data, in order to make predictions or "
        "decisions without being explicitly programmed to do so.\n\n"
        "Deep learning is part of a broader family of machine learning methods based "
        "on artificial neural networks with representation learning. Learning can be "
        "supervised, semi-supervised or unsupervised.\n\n"
        "Natural language processing (NLP) is a subfield of linguistics, computer science, "
        "and artificial intelligence concerned with the interactions between computers and "
        "human language, in particular how to program computers to process and analyze "
        "large amounts of natural language data.\n\n"
        "The ultimate goal of NLP research is to enable computers to understand language "
        "in a manner that is valuable, to the extent that they could understand what people "
        "mean when they say or write something.\n\n"
        "Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from "
        "an external knowledge base to ground large language models (LLMs) on the most "
        "accurate, up-to-date information and to give users insight into LLMs generative process."
    )


@pytest.fixture(scope="session")
def tmp_dir() -> Generator[Path, None, None]:
    d = Path(tempfile.mkdtemp(prefix="rag_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestChunking:

    def test_chunk_size_within_bounds(self, sample_text):
        from app.core.ingestion import RecursiveCharSplitter, estimate_tokens

        splitter = RecursiveCharSplitter(max_tokens=100, overlap_tokens=10)
        chunks = splitter.split(sample_text)

        assert len(chunks) > 0, "Should produce at least one chunk"
        for chunk in chunks:
            tokens = estimate_tokens(chunk)
            assert tokens <= 120, (
                f"Chunk exceeds max_tokens: {tokens} tokens in '{chunk[:50]}...'"
            )

    def test_no_empty_chunks(self, sample_text):
        from app.core.ingestion import RecursiveCharSplitter

        splitter = RecursiveCharSplitter(max_tokens=200, overlap_tokens=20)
        chunks = splitter.split(sample_text)
        for chunk in chunks:
            assert chunk.strip(), "Found empty chunk"

    def test_all_content_preserved(self, sample_text):
        from app.core.ingestion import RecursiveCharSplitter

        splitter = RecursiveCharSplitter(max_tokens=150, overlap_tokens=15)
        chunks = splitter.split(sample_text)

        combined = " ".join(chunks).lower()
        original_words = set(sample_text.lower().split())

        missing = [
            w for w in original_words
            if len(w) > 5 and w not in combined
        ]
        assert len(missing) == 0, f"Words missing from chunks: {missing[:10]}"

    def test_overlap_creates_shared_content(self, sample_text):
        from app.core.ingestion import RecursiveCharSplitter

        splitter = RecursiveCharSplitter(max_tokens=80, overlap_tokens=20)
        chunks = splitter.split(sample_text)

        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        shared_found = False
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split()[-20:])
            words_b = set(chunks[i + 1].split()[:20])
            if words_a & words_b:
                shared_found = True
                break

        assert shared_found, "No overlap detected between consecutive chunks"

    def test_chunk_metadata_correctness(self, tmp_dir):
        from app.core.ingestion import PDFIngester, DocumentChunk

        ingester = PDFIngester(chunk_size=200, chunk_overlap=20)

        chunk = DocumentChunk(
            chunk_id="test_doc_p3_c1",
            text="This is a test chunk with some content.",
            source_file="test_doc.pdf",
            page_number=3,
            chunk_index=1,
            total_chunks_in_page=5,
            char_count=38,
            token_estimate=10,
        )

        assert chunk.chunk_id == "test_doc_p3_c1"
        assert chunk.page_number == 3
        assert chunk.citation == "[test_doc.pdf, p. 3]"
        assert chunk.token_estimate > 0

    def test_text_cleaning(self):
        from app.core.ingestion import clean_text

        dirty = "The ﬁrst  step is under-\nstanding the prob- \nlem.\n\n\n\nNew para."
        cleaned = clean_text(dirty)

        assert "ﬁ" not in cleaned, "Unicode ligature should be normalised"
        assert "  " not in cleaned, "Double spaces should be collapsed"

        assert "understanding" in cleaned or "under-\nstanding" not in cleaned


class TestEmbedder:

    @pytest.fixture(scope="class")
    def embedder(self):
        from app.core.embedder import Embedder
        return Embedder(model_name="BAAI/bge-small-en-v1.5", batch_size=8)

    def test_embedding_dimension(self, embedder):
        emb = embedder.embed_query("What is machine learning?")
        assert emb.shape == (1, 384), f"Expected (1, 384), got {emb.shape}"

    def test_embedding_normalised(self, embedder):
        texts = ["Hello world", "Machine learning is great", "RAG systems are useful"]
        from app.models.schemas import DocumentChunk
        chunks = [
            DocumentChunk(
                chunk_id=f"t{i}",
                text=t,
                source_file="test.pdf",
                page_number=1,
                chunk_index=i,
                total_chunks_in_page=3,
                char_count=len(t),
                token_estimate=len(t) // 4,
            )
            for i, t in enumerate(texts)
        ]
        embs = embedder.embed_documents(chunks)

        for i, emb in enumerate(embs):
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-5, f"Embedding {i} not normalised: norm={norm}"

    def test_similar_texts_have_higher_score(self, embedder):
        q = embedder.embed_query("What is deep learning?")

        from app.models.schemas import DocumentChunk

        def make_chunk(i, text):
            return DocumentChunk(
                chunk_id=f"c{i}", text=text, source_file="t.pdf",
                page_number=1, chunk_index=i, total_chunks_in_page=2,
                char_count=len(text), token_estimate=len(text) // 4
            )

        related = embedder.embed_documents([
            make_chunk(0, "Deep learning uses neural networks for pattern recognition.")
        ])
        unrelated = embedder.embed_documents([
            make_chunk(1, "The stock market closed higher on Friday afternoon.")
        ])

        score_related = float(np.dot(q[0], related[0]))
        score_unrelated = float(np.dot(q[0], unrelated[0]))

        assert score_related > score_unrelated, (
            f"Related ({score_related:.3f}) should score higher than "
            f"unrelated ({score_unrelated:.3f})"
        )

    def test_empty_input(self, embedder):
        result = embedder.embed_documents([])
        assert result.shape == (0, 384)


class TestVectorStore:

    @pytest.fixture
    def store_and_chunks(self, tmp_dir):
        from app.core.vector_store import VectorStore
        from app.models.schemas import DocumentChunk

        store_dir = tmp_dir / "test_store"
        store = VectorStore(embedding_dim=4, index_dir=store_dir)

        chunks = []
        embeddings = []
        texts = [
            ("machine learning basics", 0),
            ("deep neural networks explained", 0),
            ("stock market analysis report", 1),
            ("financial quarterly results", 1),
            ("NLP natural language processing", 0),
        ]
        for i, (text, page) in enumerate(texts):
            chunks.append(DocumentChunk(
                chunk_id=f"doc_p{page}_c{i}",
                text=text,
                source_file="test.pdf",
                page_number=page + 1,
                chunk_index=i,
                total_chunks_in_page=3,
                char_count=len(text),
                token_estimate=len(text) // 4,
            ))

            vec = np.zeros(4, dtype=np.float32)
            vec[page] = 1.0 - i * 0.05
            vec /= np.linalg.norm(vec)
            embeddings.append(vec)

        embeddings_array = np.array(embeddings, dtype=np.float32)
        store.add(chunks, embeddings_array)
        return store, chunks, embeddings_array

    def test_search_returns_correct_count(self, store_and_chunks):
        store, chunks, embeddings = store_and_chunks
        query = embeddings[0:1]
        results = store.search(query, top_k=3, threshold=0.0)
        assert len(results) <= 3

    def test_threshold_filters_low_scores(self, store_and_chunks):
        store, chunks, embeddings = store_and_chunks
        query = embeddings[0:1]

        results_high = store.search(query, top_k=5, threshold=0.9)
        results_low = store.search(query, top_k=5, threshold=0.0)

        assert len(results_high) <= len(results_low)
        for r in results_high:
            assert r.score >= 0.9, f"Score {r.score} below threshold 0.9"

    def test_results_sorted_by_score(self, store_and_chunks):
        store, chunks, embeddings = store_and_chunks
        query = embeddings[0:1]
        results = store.search(query, top_k=5, threshold=0.0)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_empty_store_returns_empty(self, tmp_dir):
        from app.core.vector_store import VectorStore
        store = VectorStore(embedding_dim=4, index_dir=tmp_dir / "empty_store")
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = store.search(query, top_k=3, threshold=0.5)
        assert results == []

    def test_persistence(self, store_and_chunks, tmp_dir):
        store, chunks, _ = store_and_chunks
        store.save()

        from app.core.vector_store import VectorStore
        store2 = VectorStore(embedding_dim=4, index_dir=store.index_dir)

        assert store2.size == store.size, "Reloaded store has wrong size"
        assert store2.documents == store.documents, "Reloaded store has wrong docs"

    def test_mmr_reduces_redundancy(self, store_and_chunks):
        store, chunks, embeddings = store_and_chunks
        query = embeddings[0:1]

        candidates = store.search(query, top_k=5, threshold=0.0)
        mmr_result = store.mmr_rerank(candidates, query, final_k=2, lambda_=0.5)

        assert len(mmr_result) <= 2


class TestRetrievalPrecision:

    def test_precision_at_k(self):
        from app.core.vector_store import VectorStore
        from app.core.embedder import Embedder
        from app.models.schemas import DocumentChunk

        embedder = Embedder(model_name="BAAI/bge-small-en-v1.5", batch_size=8)

        ai_texts = [
            "Machine learning algorithms learn patterns from training data.",
            "Deep learning uses multiple layers of neural networks.",
            "Transformers revolutionised natural language processing tasks.",
            "Reinforcement learning trains agents through reward signals.",
            "Convolutional networks excel at image classification tasks.",
            "Attention mechanisms allow models to focus on relevant tokens.",
        ]
        finance_texts = [
            "The stock market experienced volatility in Q3 earnings.",
            "Interest rate hikes affect bond prices inversely.",
            "Portfolio diversification reduces investment risk exposure.",
            "The Federal Reserve's monetary policy impacts inflation.",
        ]

        all_texts = ai_texts + finance_texts
        all_chunks = [
            DocumentChunk(
                chunk_id=f"c{i}",
                text=t,
                source_file="corpus.pdf",
                page_number=i + 1,
                chunk_index=0,
                total_chunks_in_page=1,
                char_count=len(t),
                token_estimate=len(t) // 4,
            )
            for i, t in enumerate(all_texts)
        ]

        embeddings = embedder.embed_documents(all_chunks)

        with tempfile.TemporaryDirectory() as td:
            store = VectorStore(embedding_dim=embedder.dim, index_dir=Path(td))
            store.add(all_chunks, embeddings)

            query_emb = embedder.embed_query(
                "How do neural networks learn representations?"
            )
            results = store.search(query_emb, top_k=3, threshold=0.0)

            relevant_ids = {f"c{i}" for i in range(6)}
            retrieved_ids = {r.chunk.chunk_id for r in results}
            intersection = relevant_ids & retrieved_ids

            k = len(results)
            precision = len(intersection) / k if k > 0 else 0.0

            assert precision >= 0.6, (
                f"Precision@{k} = {precision:.2f} — below acceptable threshold 0.6. "
                f"Retrieved: {retrieved_ids}, Relevant: {relevant_ids}"
            )


class TestHallucinationGuard:

    def test_low_similarity_returns_not_found(self):
        from app.core.vector_store import VectorStore
        from app.models.schemas import DocumentChunk, QueryRequest
        from app.core.pipeline import RAGPipeline
        from app.core.config import Settings

        settings = Settings(
            similarity_threshold=0.99,
            retrieval_top_k=3,
            retrieval_final_k=2,
        )

        with tempfile.TemporaryDirectory() as td:
            settings.data_dir = Path(td) / "pdfs"
            settings.index_dir = Path(td) / "index"
            settings.model_dir = Path(td) / "models"
            settings.llm_model_path = "nonexistent.gguf"

            settings.data_dir.mkdir(parents=True)

            pipeline = RAGPipeline(settings)

            from app.core.embedder import Embedder
            embedder = Embedder()
            chunk = DocumentChunk(
                chunk_id="c1",
                text="The capital of France is Paris.",
                source_file="geo.pdf",
                page_number=1,
                chunk_index=0,
                total_chunks_in_page=1,
                char_count=30,
                token_estimate=8,
            )
            emb = embedder.embed_documents([chunk])
            pipeline.vector_store.add([chunk], emb)

            response = pipeline.query(QueryRequest(
                question="What is the chemical formula for water?"
            ))

            assert not response.found_in_documents, (
                "System should return found_in_documents=False for out-of-scope query"
            )

    def test_empty_index_returns_not_found(self):
        from app.core.vector_store import VectorStore
        from app.models.schemas import QueryRequest
        from app.core.pipeline import RAGPipeline
        from app.core.config import Settings

        with tempfile.TemporaryDirectory() as td:
            settings = Settings(
                similarity_threshold=0.5,
                llm_model_path="nonexistent.gguf",
            )
            settings.data_dir = Path(td) / "pdfs"
            settings.index_dir = Path(td) / "index"
            settings.model_dir = Path(td) / "models"
            settings.data_dir.mkdir(parents=True)

            pipeline = RAGPipeline(settings)

            results = pipeline.vector_store.search(
                query_embedding=np.zeros((1, settings.embedding_dim), dtype=np.float32),
                top_k=3,
                threshold=0.5,
            )
            assert results == [], "Empty store should return empty results"


class TestPromptConstruction:

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        return "\n\n".join(m["content"] for m in messages)

    def test_prompt_contains_question(self):
        from app.core.llm import build_messages
        from app.models.schemas import DocumentChunk, RetrievedChunk

        chunk = DocumentChunk(
            chunk_id="c1", text="AI is transforming industries.",
            source_file="ai.pdf", page_number=2, chunk_index=0,
            total_chunks_in_page=1, char_count=30, token_estimate=8,
        )
        rc = RetrievedChunk(chunk=chunk, score=0.85)

        messages = build_messages(
            question="How is AI used?",
            retrieved_chunks=[rc],
            system_prompt="Answer only from context.",
        )

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Answer only from context."

        prompt = self._flatten(messages)
        assert "How is AI used?" in prompt
        assert "ai.pdf" in prompt
        assert "Página 2" in prompt
        assert "AI is transforming industries." in prompt

    def test_prompt_with_no_chunks(self):
        from app.core.llm import build_messages

        messages = build_messages(
            question="What is X?",
            retrieved_chunks=[],
            system_prompt="Answer only from context.",
        )

        assert "Nenhum contexto relevante encontrado" in self._flatten(messages)

    def test_multiple_sources_in_prompt(self):
        from app.core.llm import build_messages
        from app.models.schemas import DocumentChunk, RetrievedChunk

        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=f"c{i}", text=f"Content from doc {i}.",
                    source_file=f"doc{i}.pdf", page_number=i + 1,
                    chunk_index=0, total_chunks_in_page=1,
                    char_count=20, token_estimate=5,
                ),
                score=0.9 - i * 0.1,
            )
            for i in range(3)
        ]

        prompt = self._flatten(build_messages("Test question", chunks, "Be precise."))
        for i in range(3):
            assert f"doc{i}.pdf" in prompt, f"Source doc{i}.pdf missing from prompt"


class TestAPI:

    @pytest.fixture(scope="class")
    def client(self, tmp_dir):
        from app.main import app
        from app.core.pipeline import RAGPipeline
        from app.models.schemas import (
            QueryResponse,
            IngestResponse,
            IndexStatsResponse,
        )

        mock_pipeline = MagicMock(spec=RAGPipeline)
        mock_pipeline.vector_store = MagicMock()
        mock_pipeline.vector_store.size = 5

        mock_pipeline.query.return_value = QueryResponse(
            question="Test question",
            answer="Test answer based on documents.",
            retrieved_chunks=[],
            full_prompt="[INST] Test prompt [/INST]",
            found_in_documents=True,
            latency_ms=123.4,
        )

        mock_pipeline.ingest.return_value = IngestResponse(
            chunks_indexed=42,
            documents_processed=3,
            latency_ms=1500.0,
        )

        mock_pipeline.get_stats.return_value = IndexStatsResponse(
            total_chunks=42,
            documents=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
            index_type="FlatIP (exact cosine)",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
        )
        with patch("app.main.RAGPipeline", return_value=mock_pipeline), \
             patch("app.main.MemoryOrchestrator", side_effect=RuntimeError("disabled in tests")):
            with TestClient(app) as c:
                yield c

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_query_endpoint_returns_200(self, client):
        response = client.post(
            "/api/v1/query",
            json={"question": "What is machine learning?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "retrieved_chunks" in data
        assert "full_prompt" in data
        assert "found_in_documents" in data

    def test_query_validation_min_length(self, client):
        response = client.post("/api/v1/query", json={"question": "AI"})
        assert response.status_code == 422

    def test_ingest_endpoint(self, client):
        response = client.post(
            "/api/v1/ingest",
            json={"force_reindex": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "chunks_indexed" in data

    def test_stats_endpoint(self, client):
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_chunks"] == 42
        assert len(data["documents"]) == 3
