import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.models.schemas import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

_INDEX_FILE = "faiss.index"
_META_FILE = "chunks_metadata.json"


class VectorStore:
    """
    FAISS vector store that maps embedding vectors ↔ DocumentChunk objects.
    """

    def __init__(self, embedding_dim: int, index_dir: Path):
        self.dim = embedding_dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Parallel lists: index position i → chunks[i]
        self._chunks: list[DocumentChunk] = []
        self._index: Optional[faiss.IndexFlatIP] = None

        # Try to load an existing index
        if self._persisted():
            self._load()
        else:
            self._create_index()

    def _create_index(self) -> None:
        """
        Create a fresh FlatIP (inner-product) index.

        Because embeddings are L2-normalised, inner-product == cosine similarity.
        FlatIP is exact — no approximation error.
        """
        self._index = faiss.IndexFlatIP(self.dim)
        logger.info(f"Created new FlatIP FAISS index (dim={self.dim})")

    def add(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        chunks     : list of N DocumentChunk objects
        embeddings : float32 array of shape (N, dim), must be L2-normalised
        """
        assert len(chunks) == len(embeddings), "Chunk / embedding count mismatch"
        assert embeddings.dtype == np.float32, "Embeddings must be float32"

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self._chunks)}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: float,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top_k most similar chunks above `threshold`.

        Returns a list of RetrievedChunk sorted by descending score,
        with all scores below `threshold` discarded.
        """
        if self.size == 0:
            logger.warning("Search called on empty index")
            return []

        # Ensure query is 2-D (1, dim) for FAISS
        q = query_embedding.reshape(1, -1).astype(np.float32)

        k = min(top_k, self.size)
        scores, indices = self._index.search(q, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots
            if float(score) < threshold:
                continue  # below similarity threshold — discard
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(score),
                )
            )

        logger.debug(
            f"Search returned {len(results)} chunks "
            f"(threshold={threshold}, top_k={top_k})"
        )
        return results

    def mmr_rerank(
        self,
        candidates: list[RetrievedChunk],
        query_embedding: np.ndarray,
        final_k: int,
        lambda_: float = 0.6,
    ) -> list[RetrievedChunk]:
        """
        Maximal Marginal Relevance re-ranking.

        Iteratively selects the chunk that maximises:
            score = λ · sim(chunk, query) − (1−λ) · max_{s∈selected} sim(chunk, s)

        This trades a little relevance for much better diversity, avoiding
        near-duplicate chunks in the final context.

        lambda_=1.0 → pure relevance (same as top-k)
        lambda_=0.0 → pure diversity
        lambda_=0.6 → recommended default (Carbonell & Goldstein, 1998)
        """
        if not candidates:
            return []

        if len(candidates) <= final_k:
            return candidates

        # Gather embeddings for all candidates (re-embed from index)
        # We store embeddings separately for MMR to avoid a second FAISS search
        candidate_embeddings = self._get_embeddings_for_chunks(candidates)
        q = query_embedding.reshape(-1)  # (dim,)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_indices) < final_k and remaining:
            best_idx = None
            best_score = -float("inf")

            for i in remaining:
                relevance = float(np.dot(candidate_embeddings[i], q))

                if not selected_indices:
                    redundancy = 0.0
                else:
                    # Max similarity to any already-selected chunk
                    redundancy = max(
                        float(np.dot(candidate_embeddings[i], candidate_embeddings[j]))
                        for j in selected_indices
                    )

                mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def _get_embeddings_for_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> list[np.ndarray]:
        id_to_pos = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        embeddings = []
        for rc in chunks:
            pos = id_to_pos.get(rc.chunk.chunk_id)
            if pos is None:
                embeddings.append(np.zeros(self.dim, dtype=np.float32))
                continue
            embeddings.append(self._index.reconstruct(pos))
        return embeddings

    def save(self) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        index_path = self.index_dir / _INDEX_FILE
        meta_path = self.index_dir / _META_FILE

        faiss.write_index(self._index, str(index_path))

        # Serialise chunk list as JSON for human-readability / debuggability
        meta = [chunk.model_dump() for chunk in self._chunks]
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        logger.info(f"Index saved → {index_path} ({self.size} vectors)")

    def _load(self) -> None:
        """Load a previously persisted index from disk."""
        index_path = self.index_dir / _INDEX_FILE
        meta_path = self.index_dir / _META_FILE

        self._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text())
        self._chunks = [DocumentChunk(**m) for m in meta]

        logger.info(f"Index loaded ← {index_path} ({self.size} vectors)")

    def _persisted(self) -> bool:
        """Return True if a saved index exists on disk."""
        return (
            (self.index_dir / _INDEX_FILE).exists()
            and (self.index_dir / _META_FILE).exists()
        )

    def clear(self) -> None:
        """Delete the in-memory index and metadata."""
        self._chunks = []
        self._create_index()
        logger.info("Index cleared")

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def documents(self) -> list[str]:
        """Unique source filenames present in the index."""
        return sorted({c.source_file for c in self._chunks})
