import os
import tempfile
from pathlib import Path

import pytest

from app.core.config import Settings


def _default_settings() -> Settings:
    """Settings sem carregar o .env local — testa os defaults da classe."""
    return Settings(_env_file=None)


class TestSettingsDefaults:

    def test_settings_default_data_dir(self) -> None:
        assert _default_settings().data_dir == Path("data/pdfs")

    def test_settings_default_index_dir(self) -> None:
        assert _default_settings().index_dir == Path("data/index")

    def test_settings_default_model_dir(self) -> None:
        assert _default_settings().model_dir == Path("models")

    def test_settings_default_chunk_size(self) -> None:
        assert _default_settings().chunk_size == 300

    def test_settings_default_chunk_overlap(self) -> None:
        assert _default_settings().chunk_overlap == 50

    def test_settings_default_embedding_model(self) -> None:
        assert _default_settings().embedding_model == "intfloat/multilingual-e5-small"

    def test_settings_default_embedding_dim(self) -> None:
        assert _default_settings().embedding_dim == 384

    def test_settings_default_embedding_batch_size(self) -> None:
        assert _default_settings().embedding_batch_size == 128

    def test_settings_default_retrieval_top_k(self) -> None:
        assert _default_settings().retrieval_top_k == 20

    def test_settings_default_retrieval_final_k(self) -> None:
        assert _default_settings().retrieval_final_k == 5

    def test_settings_default_similarity_threshold(self) -> None:
        assert _default_settings().similarity_threshold == 0.50

    def test_settings_default_mmr_lambda(self) -> None:
        assert _default_settings().mmr_lambda == 0.75

    def test_settings_default_llm_context_length(self) -> None:
        assert _default_settings().llm_context_length == 4096

    def test_settings_default_llm_max_new_tokens(self) -> None:
        assert _default_settings().llm_max_new_tokens == 512

    def test_settings_default_llm_temperature(self) -> None:
        assert _default_settings().llm_temperature == 0.1

    def test_settings_default_llm_n_threads(self) -> None:
        assert _default_settings().llm_n_threads == 8

    def test_settings_default_use_chroma(self) -> None:
        assert _default_settings().use_chroma is False


class TestSettingsValidation:

    def test_similarity_threshold_valid(self) -> None:
        settings = Settings(similarity_threshold=0.5)
        assert settings.similarity_threshold == 0.5

    def test_similarity_threshold_at_boundaries(self) -> None:
        settings_min = Settings(similarity_threshold=0.0)
        settings_max = Settings(similarity_threshold=1.0)
        assert settings_min.similarity_threshold == 0.0
        assert settings_max.similarity_threshold == 1.0

    def test_similarity_threshold_invalid_too_high(self) -> None:
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            Settings(similarity_threshold=1.5)

    def test_similarity_threshold_invalid_too_low(self) -> None:
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            Settings(similarity_threshold=-0.1)

    def test_mmr_lambda_valid(self) -> None:
        settings = Settings(mmr_lambda=0.7)
        assert settings.mmr_lambda == 0.7

    def test_mmr_lambda_at_boundaries(self) -> None:
        settings_min = Settings(mmr_lambda=0.0)
        settings_max = Settings(mmr_lambda=1.0)
        assert settings_min.mmr_lambda == 0.0
        assert settings_max.mmr_lambda == 1.0

    def test_mmr_lambda_invalid_too_high(self) -> None:
        with pytest.raises(ValueError, match="mmr_lambda must be in"):
            Settings(mmr_lambda=1.5)

    def test_mmr_lambda_invalid_too_low(self) -> None:
        with pytest.raises(ValueError, match="mmr_lambda must be in"):
            Settings(mmr_lambda=-0.1)


class TestSettingsEnvironmentVariables:

    def test_chunk_size_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "256")
        settings = Settings()
        assert settings.chunk_size == 256

    def test_chunk_overlap_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("CHUNK_OVERLAP", "32")
        settings = Settings()
        assert settings.chunk_overlap == 32

    def test_similarity_threshold_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.5")
        settings = Settings()
        assert settings.similarity_threshold == 0.5

    def test_embedding_batch_size_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "64")
        settings = Settings()
        assert settings.embedding_batch_size == 64

    def test_retrieval_top_k_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
        settings = Settings()
        assert settings.retrieval_top_k == 10


class TestSettingsPaths:

    def test_data_dir_is_path_object(self) -> None:
        settings = Settings()
        assert isinstance(settings.data_dir, Path)

    def test_index_dir_is_path_object(self) -> None:
        settings = Settings()
        assert isinstance(settings.index_dir, Path)

    def test_model_dir_is_path_object(self) -> None:
        settings = Settings()
        assert isinstance(settings.model_dir, Path)

    def test_custom_data_dir_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("DATA_DIR", "/custom/pdfs")
        settings = Settings()
        assert settings.data_dir == Path("/custom/pdfs")

    def test_custom_index_dir_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("INDEX_DIR", "/custom/index")
        settings = Settings()
        assert settings.index_dir == Path("/custom/index")
