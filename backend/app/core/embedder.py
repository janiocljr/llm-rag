"""
Embedding module for semantic text representation.

This module provides sentence embeddings using HuggingFace models
(primarily BAAI/bge-m3 for multilingual support).
"""

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX = "search_query: "
_DOCUMENT_PREFIX = "search_document: "


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    """
    Wraps SentenceTransformer to produce L2-normalized embeddings.

    The same instance handles both document indexing and query encoding
    to ensure embedding space compatibility.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 64,
        device: str = "auto",
    ) -> None:
        """Initialize the embedder with a specific model and device.

        Args:
            model_name: HuggingFace model ID
            batch_size: Batch size for embedding generation
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        resolved_device = _resolve_device(device)
        logger.info(f"Loading embedding model: {model_name} (device={resolved_device})")
        self.model_name = model_name
        self.batch_size = batch_size

        self._model = SentenceTransformer(
            model_name,
            device=resolved_device,
        )
        self.dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready — dim={self.dim}, device={resolved_device}")


    def embed_documents(self, chunks: list[DocumentChunk]) -> np.ndarray:
        """
        Embed document chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Float32 array of shape (N, dim), L2-normalized
        """
        texts = [chunk.text for chunk in chunks]
        return self._encode(texts, is_query=False)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.

        Args:
            query: Query text

        Returns:
            Float32 array of shape (1, dim), L2-normalized
        """
        return self._encode([query], is_query=True)


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
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
