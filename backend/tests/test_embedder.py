"""
Unit tests for app.core.embedder module.

Tests cover:
- Device resolution (CPU/CUDA/MPS)
- Embedder initialization
- Document embedding
- Query embedding
- Edge cases (empty inputs, etc.)
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.core.embedder import Embedder, _resolve_device
from app.models.schemas import DocumentChunk


class TestResolveDevice:
    """Tests for device resolution utility."""

    def test_resolve_device_explicit_cpu(self) -> None:
        """Test that explicit 'cpu' is returned as-is."""
        assert _resolve_device("cpu") == "cpu"

    def test_resolve_device_explicit_cuda(self) -> None:
        """Test that explicit 'cuda' is returned as-is."""
        assert _resolve_device("cuda") == "cuda"

    def test_resolve_device_explicit_mps(self) -> None:
        """Test that explicit 'mps' is returned as-is."""
        assert _resolve_device("mps") == "mps"

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_resolve_device_auto_prefers_mps(self, mock_cuda, mock_mps) -> None:
        """Test that 'auto' prefers MPS when available."""
        mock_mps.return_value = True
        mock_cuda.return_value = True
        assert _resolve_device("auto") == "mps"

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_resolve_device_auto_prefers_cuda_over_cpu(self, mock_cuda, mock_mps) -> None:
        """Test that 'auto' prefers CUDA when MPS unavailable."""
        mock_mps.return_value = False
        mock_cuda.return_value = True
        assert _resolve_device("auto") == "cuda"

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_resolve_device_auto_falls_back_to_cpu(self, mock_cuda, mock_mps) -> None:
        """Test that 'auto' falls back to CPU when GPU unavailable."""
        mock_mps.return_value = False
        mock_cuda.return_value = False
        assert _resolve_device("auto") == "cpu"


class TestEmbedder:
    """Tests for Embedder class."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        with patch("app.core.embedder.SentenceTransformer") as mock:
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 1024
            model.encode.return_value = np.random.randn(1, 1024).astype(np.float32)
            mock.return_value = model
            yield mock

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_embedder_initialization(self, mock_resolve, mock_sentence_transformer) -> None:
        """Test Embedder initialization."""
        embedder = Embedder(model_name="BAAI/bge-m3", batch_size=64, device="auto")
        assert embedder.model_name == "BAAI/bge-m3"
        assert embedder.batch_size == 64
        assert embedder.dim == 1024

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_embed_documents(self, mock_resolve, mock_sentence_transformer) -> None:
        """Test embedding a list of documents."""
        embedder = Embedder()
        chunks = [
            DocumentChunk(
                chunk_id="test_1",
                text="First document",
                source_file="doc.pdf",
                page_number=1,
                chunk_index=0,
                total_chunks_in_page=2,
                char_count=15,
                token_estimate=4,
            ),
            DocumentChunk(
                chunk_id="test_2",
                text="Second document",
                source_file="doc.pdf",
                page_number=1,
                chunk_index=1,
                total_chunks_in_page=2,
                char_count=16,
                token_estimate=4,
            ),
        ]

        with patch.object(embedder, "_encode") as mock_encode:
            mock_encode.return_value = np.zeros((2, 1024), dtype=np.float32)
            result = embedder.embed_documents(chunks)
            assert result.shape == (2, 1024)
            assert mock_encode.called

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_embed_query(self, mock_resolve, mock_sentence_transformer) -> None:
        """Test embedding a query string."""
        embedder = Embedder()

        with patch.object(embedder, "_encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 1024), dtype=np.float32)
            result = embedder.embed_query("test query")
            assert result.shape == (1, 1024)
            assert mock_encode.called

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_empty_texts(self, mock_resolve, mock_sentence_transformer) -> None:
        """Test encoding empty text list."""
        embedder = Embedder()
        result = embedder._encode([], is_query=False)
        assert result.shape == (0, 1024)
        assert result.dtype == np.float32

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_with_query_prefix(self, mock_resolve, mock_sentence_transformer) -> None:
        """Test that query prefix is added when is_query=True."""
        embedder = Embedder()
        texts = ["test query"]
        with patch.object(embedder._model, "encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 1024), dtype=np.float32)
            embedder._encode(texts, is_query=True)

            called_texts = mock_encode.call_args[0][0]
            assert "Represent this sentence" in called_texts[0]

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_returns_float32(self, mock_resolve, mock_sentence_transformer) -> None:
        """Test that embeddings are returned as float32."""
        embedder = Embedder()
        with patch.object(embedder._model, "encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 1024), dtype=np.float64)
            result = embedder._encode(["test"], is_query=False)
            assert result.dtype == np.float32

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_batch_size(self, mock_resolve, mock_sentence_transformer) -> None:
        """Test that batch size is passed to model."""
        embedder = Embedder(batch_size=32)
        with patch.object(embedder._model, "encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 1024), dtype=np.float32)
            embedder._encode(["test"], is_query=False)

            assert mock_encode.call_args[1]["batch_size"] == 32
