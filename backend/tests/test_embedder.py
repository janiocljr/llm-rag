import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.core.embedder import (
    Embedder,
    _resolve_device,
    _resolve_prefixes,
    _BGE_EN_QUERY_PREFIX,
    _E5_DOCUMENT_PREFIX,
    _E5_QUERY_PREFIX,
)
from app.models.schemas import DocumentChunk


class TestResolveDevice:

    def test_resolve_device_explicit_cpu(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_resolve_device_explicit_cuda(self) -> None:
        assert _resolve_device("cuda") == "cuda"

    def test_resolve_device_explicit_mps(self) -> None:
        assert _resolve_device("mps") == "mps"

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_resolve_device_auto_prefers_mps(self, mock_cuda, mock_mps) -> None:
        mock_mps.return_value = True
        mock_cuda.return_value = True
        assert _resolve_device("auto") == "mps"

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_resolve_device_auto_prefers_cuda_over_cpu(self, mock_cuda, mock_mps) -> None:
        mock_mps.return_value = False
        mock_cuda.return_value = True
        assert _resolve_device("auto") == "cuda"

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_resolve_device_auto_falls_back_to_cpu(self, mock_cuda, mock_mps) -> None:
        mock_mps.return_value = False
        mock_cuda.return_value = False
        assert _resolve_device("auto") == "cpu"


class TestResolvePrefixes:
    def test_e5_models_use_query_and_passage_prefixes(self) -> None:
        assert _resolve_prefixes("intfloat/multilingual-e5-small") == (
            _E5_QUERY_PREFIX,
            _E5_DOCUMENT_PREFIX,
        )
        assert _resolve_prefixes("intfloat/e5-large-v2") == (
            _E5_QUERY_PREFIX,
            _E5_DOCUMENT_PREFIX,
        )

    def test_bge_en_uses_instruction_prefix_on_query_only(self) -> None:
        assert _resolve_prefixes("BAAI/bge-large-en-v1.5") == (_BGE_EN_QUERY_PREFIX, "")

    def test_bge_m3_uses_no_prefixes(self) -> None:
        assert _resolve_prefixes("BAAI/bge-m3") == ("", "")

    def test_unknown_model_uses_no_prefixes(self) -> None:
        assert _resolve_prefixes("sentence-transformers/all-MiniLM-L6-v2") == ("", "")


class TestEmbedder:

    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("app.core.embedder.SentenceTransformer") as mock:
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384
            model.encode.return_value = np.random.default_rng(1).random((1, 384)).astype(np.float32)
            mock.return_value = model
            yield mock

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_embedder_initialization(self, mock_resolve, mock_sentence_transformer) -> None:
        embedder = Embedder(model_name="intfloat/multilingual-e5-small", batch_size=64, device="auto")
        assert embedder.model_name == "intfloat/multilingual-e5-small"
        assert embedder.batch_size == 64
        assert embedder.dim == 384
        assert embedder.query_prefix == _E5_QUERY_PREFIX
        assert embedder.document_prefix == _E5_DOCUMENT_PREFIX

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_embed_documents(self, mock_resolve, mock_sentence_transformer) -> None:
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
            mock_encode.return_value = np.zeros((2, 384), dtype=np.float32)
            result = embedder.embed_documents(chunks)
            assert result.shape == (2, 384)
            mock_encode.assert_called_once_with(
                ["First document", "Second document"], prefix=_E5_DOCUMENT_PREFIX
            )

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_embed_query(self, mock_resolve, mock_sentence_transformer) -> None:
        embedder = Embedder()

        with patch.object(embedder, "_encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 384), dtype=np.float32)
            result = embedder.embed_query("test query")
            assert result.shape == (1, 384)
            mock_encode.assert_called_once_with(["test query"], prefix=_E5_QUERY_PREFIX)

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_empty_texts(self, mock_resolve, mock_sentence_transformer) -> None:
        embedder = Embedder()
        result = embedder._encode([], prefix=_E5_DOCUMENT_PREFIX)
        assert result.shape == (0, 384)
        assert result.dtype == np.float32

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_applies_e5_query_prefix(self, mock_resolve, mock_sentence_transformer) -> None:
        embedder = Embedder(model_name="intfloat/multilingual-e5-small")
        with patch.object(embedder._model, "encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 384), dtype=np.float32)
            embedder._encode(["test query"], prefix=embedder.query_prefix)

            called_texts = mock_encode.call_args[0][0]
            assert called_texts[0] == "query: test query"

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_applies_e5_document_prefix(self, mock_resolve, mock_sentence_transformer) -> None:
        embedder = Embedder(model_name="intfloat/multilingual-e5-small")
        with patch.object(embedder._model, "encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 384), dtype=np.float32)
            embedder._encode(["some passage"], prefix=embedder.document_prefix)

            called_texts = mock_encode.call_args[0][0]
            assert called_texts[0] == "passage: some passage"

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_returns_float32(self, mock_resolve, mock_sentence_transformer) -> None:
        embedder = Embedder()
        with patch.object(embedder._model, "encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 384), dtype=np.float64)
            result = embedder._encode(["test"])
            assert result.dtype == np.float32

    @patch("app.core.embedder._resolve_device", return_value="cpu")
    def test_encode_batch_size(self, mock_resolve, mock_sentence_transformer) -> None:
        embedder = Embedder(batch_size=32)
        with patch.object(embedder._model, "encode") as mock_encode:
            mock_encode.return_value = np.zeros((1, 384), dtype=np.float32)
            embedder._encode(["test"])

            assert mock_encode.call_args[1]["batch_size"] == 32
