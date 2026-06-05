"""
Unit tests for app.core.config module.

Tests cover:
- Settings initialization with defaults
- Settings validation
- Environment variable overrides
- Path creation
"""

import os
import tempfile
from pathlib import Path

import pytest

from app.core.config import Settings


class TestSettingsDefaults:
    """Test Settings default values."""

    def test_settings_default_data_dir(self) -> None:
        """Test default data directory."""
        settings = Settings()
        assert settings.data_dir == Path("data/pdfs")

    def test_settings_default_index_dir(self) -> None:
        """Test default index directory."""
        settings = Settings()
        assert settings.index_dir == Path("data/index")

    def test_settings_default_model_dir(self) -> None:
        """Test default model directory."""
        settings = Settings()
        assert settings.model_dir == Path("models")

    def test_settings_default_chunk_size(self) -> None:
        """Test default chunk size."""
        settings = Settings()
        assert settings.chunk_size == 512

    def test_settings_default_chunk_overlap(self) -> None:
        """Test default chunk overlap."""
        settings = Settings()
        assert settings.chunk_overlap == 64

    def test_settings_default_embedding_model(self) -> None:
        """Test default embedding model."""
        settings = Settings()
        assert settings.embedding_model == "BAAI/bge-m3"

    def test_settings_default_embedding_dim(self) -> None:
        """Test default embedding dimension."""
        settings = Settings()
        assert settings.embedding_dim == 1024

    def test_settings_default_embedding_batch_size(self) -> None:
        """Test default embedding batch size."""
        settings = Settings()
        assert settings.embedding_batch_size == 128

    def test_settings_default_retrieval_top_k(self) -> None:
        """Test default retrieval top-k."""
        settings = Settings()
        assert settings.retrieval_top_k == 5

    def test_settings_default_retrieval_final_k(self) -> None:
        """Test default retrieval final-k."""
        settings = Settings()
        assert settings.retrieval_final_k == 3

    def test_settings_default_similarity_threshold(self) -> None:
        """Test default similarity threshold."""
        settings = Settings()
        assert settings.similarity_threshold == 0.35

    def test_settings_default_mmr_lambda(self) -> None:
        """Test default MMR lambda."""
        settings = Settings()
        assert settings.mmr_lambda == 0.6

    def test_settings_default_llm_context_length(self) -> None:
        """Test default LLM context length."""
        settings = Settings()
        assert settings.llm_context_length == 4096

    def test_settings_default_llm_max_new_tokens(self) -> None:
        """Test default LLM max new tokens."""
        settings = Settings()
        assert settings.llm_max_new_tokens == 512

    def test_settings_default_llm_temperature(self) -> None:
        """Test default LLM temperature."""
        settings = Settings()
        assert settings.llm_temperature == 0.1

    def test_settings_default_llm_n_threads(self) -> None:
        """Test default LLM thread count."""
        settings = Settings()
        assert settings.llm_n_threads == 8

    def test_settings_default_use_chroma(self) -> None:
        """Test default Chroma setting."""
        settings = Settings()
        assert settings.use_chroma is False


class TestSettingsValidation:
    """Test Settings field validation."""

    def test_similarity_threshold_valid(self) -> None:
        """Test that valid similarity thresholds pass."""
        settings = Settings(similarity_threshold=0.5)
        assert settings.similarity_threshold == 0.5

    def test_similarity_threshold_at_boundaries(self) -> None:
        """Test similarity threshold at valid boundaries."""
        settings_min = Settings(similarity_threshold=0.0)
        settings_max = Settings(similarity_threshold=1.0)
        assert settings_min.similarity_threshold == 0.0
        assert settings_max.similarity_threshold == 1.0

    def test_similarity_threshold_invalid_too_high(self) -> None:
        """Test that similarity threshold > 1.0 raises error."""
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            Settings(similarity_threshold=1.5)

    def test_similarity_threshold_invalid_too_low(self) -> None:
        """Test that similarity threshold < 0.0 raises error."""
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            Settings(similarity_threshold=-0.1)

    def test_mmr_lambda_valid(self) -> None:
        """Test that valid MMR lambda values pass."""
        settings = Settings(mmr_lambda=0.7)
        assert settings.mmr_lambda == 0.7

    def test_mmr_lambda_at_boundaries(self) -> None:
        """Test MMR lambda at valid boundaries."""
        settings_min = Settings(mmr_lambda=0.0)
        settings_max = Settings(mmr_lambda=1.0)
        assert settings_min.mmr_lambda == 0.0
        assert settings_max.mmr_lambda == 1.0

    def test_mmr_lambda_invalid_too_high(self) -> None:
        """Test that MMR lambda > 1.0 raises error."""
        with pytest.raises(ValueError, match="mmr_lambda must be in"):
            Settings(mmr_lambda=1.5)

    def test_mmr_lambda_invalid_too_low(self) -> None:
        """Test that MMR lambda < 0.0 raises error."""
        with pytest.raises(ValueError, match="mmr_lambda must be in"):
            Settings(mmr_lambda=-0.1)


class TestSettingsEnvironmentVariables:
    """Test Settings environment variable overrides."""

    def test_chunk_size_from_env(self, monkeypatch) -> None:
        """Test chunk size can be set via environment variable."""
        monkeypatch.setenv("CHUNK_SIZE", "256")
        settings = Settings()
        assert settings.chunk_size == 256

    def test_chunk_overlap_from_env(self, monkeypatch) -> None:
        """Test chunk overlap can be set via environment variable."""
        monkeypatch.setenv("CHUNK_OVERLAP", "32")
        settings = Settings()
        assert settings.chunk_overlap == 32

    def test_similarity_threshold_from_env(self, monkeypatch) -> None:
        """Test similarity threshold can be set via environment variable."""
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.5")
        settings = Settings()
        assert settings.similarity_threshold == 0.5

    def test_embedding_batch_size_from_env(self, monkeypatch) -> None:
        """Test embedding batch size can be set via environment variable."""
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "64")
        settings = Settings()
        assert settings.embedding_batch_size == 64

    def test_retrieval_top_k_from_env(self, monkeypatch) -> None:
        """Test retrieval top-k can be set via environment variable."""
        monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
        settings = Settings()
        assert settings.retrieval_top_k == 10


class TestSettingsPaths:
    """Test Settings path handling."""

    def test_data_dir_is_path_object(self) -> None:
        """Test that data_dir is a Path object."""
        settings = Settings()
        assert isinstance(settings.data_dir, Path)

    def test_index_dir_is_path_object(self) -> None:
        """Test that index_dir is a Path object."""
        settings = Settings()
        assert isinstance(settings.index_dir, Path)

    def test_model_dir_is_path_object(self) -> None:
        """Test that model_dir is a Path object."""
        settings = Settings()
        assert isinstance(settings.model_dir, Path)

    def test_custom_data_dir_from_env(self, monkeypatch) -> None:
        """Test custom data directory from environment variable."""
        monkeypatch.setenv("DATA_DIR", "/custom/pdfs")
        settings = Settings()
        assert settings.data_dir == Path("/custom/pdfs")

    def test_custom_index_dir_from_env(self, monkeypatch) -> None:
        """Test custom index directory from environment variable."""
        monkeypatch.setenv("INDEX_DIR", "/custom/index")
        settings = Settings()
        assert settings.index_dir == Path("/custom/index")
