import logging
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX = "search_query: "
_DOCUMENT_PREFIX = "search_document: "


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _setup_hf_env() -> None:
    hf_home = Path.home() / ".cache" / "huggingface"
    cache_dir = hf_home / "hub"

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if "HF_HUB_OFFLINE" not in os.environ and "REQUESTS_CA_BUNDLE" not in os.environ:
        has_cached_models = cache_dir.exists() and any(cache_dir.glob("models--*"))
        if has_cached_models:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ["REQUESTS_CA_BUNDLE"] = ""
            os.environ["CURL_CA_BUNDLE"] = ""


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 64,
        device: str = "auto",
    ) -> None:
        _setup_hf_env()
        resolved_device = _resolve_device(device)
        logger.info(f"Loading embedding model: {model_name} (device={resolved_device})")
        self.model_name = model_name
        self.batch_size = batch_size

        try:
            self._model = SentenceTransformer(model_name, device=resolved_device)
        except Exception:
            logger.exception("Failed to load embedding model")
            raise

        self.dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready — dim={self.dim}, device={resolved_device}")

    def embed_documents(self, chunks: list[DocumentChunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        return self._encode(texts, is_query=False)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query], is_query=True)

    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
