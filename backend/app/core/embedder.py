
import logging
from typing import Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)

# BGE instruction prefix (applied to queries only)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX = "search_query: "
_DOCUMENT_PREFIX = "search_document: "

def _resolve_device(device: str) -> str:
    """Resolve 'auto' para o melhor device disponível."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class Embedder:
    """
    Wraps SentenceTransformer to produce L2-normalised embeddings.

    The same instance is used for both document indexing and query encoding
    to guarantee embedding-space compatibility.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 64,
        device: str = "auto",
    ):
        resolved_device = _resolve_device(device)
        logger.info(f"Loading embedding model: {model_name} (device={resolved_device})")
        self.model_name = model_name
        self.batch_size = batch_size

        self._model = SentenceTransformer(
            model_name,
            device=resolved_device,
        )
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready — dim={self.dim}, device={resolved_device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, chunks: list[DocumentChunk]) -> np.ndarray:
        """
        Embed a list of DocumentChunk objects.

        Returns: float32 array of shape (N, dim), L2-normalised.
        """
        texts = [chunk.text for chunk in chunks]
        return self._encode(texts, is_query=False)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns: float32 array of shape (1, dim), L2-normalised.
        Note: BGE query prefix is applied automatically.
        """
        return self._encode([query], is_query=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        """
        Core encoding method.

        is_query=True  → prepend BGE query prefix
        is_query=False → encode as-is (document mode)
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # L2 normalise → cosine via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
