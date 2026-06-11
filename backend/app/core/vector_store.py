import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.models.schemas import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

_INDEX_FILE = "faiss.index"
_META_FILE = "chunks_metadata.json"


class VectorStore:
    def __init__(self, embedding_dim: int, index_dir: Path):
        self.dim = embedding_dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._chunks: list[DocumentChunk] = []
        self._index: Optional[faiss.IndexFlatIP] = None

        if self._persisted():
            self._load()
        else:
            self._create_index()

    def _create_index(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        logger.info(f"Created new FlatIP FAISS index (dim={self.dim})")

    def add(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
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
        if self.size == 0:
            logger.warning("Search called on empty index")
            return []

        q = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.size)
        scores, indices = self._index.search(q, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < threshold:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))

        return results

    def score_at(self, position: int, query_embedding: np.ndarray) -> float:
        """Similaridade coseno entre a query e o vetor na posição dada."""
        emb = self._index.reconstruct(int(position))
        return float(np.dot(emb, query_embedding.reshape(-1)))

    def mmr_rerank(
        self,
        candidates: list[RetrievedChunk],
        query_embedding: np.ndarray,
        final_k: int,
        lambda_: float = 0.6,
        relevance_scores: Optional[list[float]] = None,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if len(candidates) <= final_k:
            return candidates

        candidate_embeddings = self._get_embeddings_for_chunks(candidates)
        q = query_embedding.reshape(-1)

        # Modelos como o e5 comprimem os cosenos num intervalo estreito
        # (~0.75-0.90 para todo o corpus). Sem normalização, o termo de
        # redundância domina as diferenças minúsculas de relevância e o MMR
        # passa a descartar exatamente os chunks mais relevantes. Min-max
        # sobre os candidatos devolve peso real ao termo de relevância.
        # relevance_scores permite usar um ranking externo (ex.: fusão RRF
        # da busca híbrida) como termo de relevância.
        if relevance_scores is not None:
            raw_relevances = np.asarray(relevance_scores, dtype=np.float64)
        else:
            raw_relevances = np.array(
                [float(np.dot(emb, q)) for emb in candidate_embeddings]
            )
        rel_min, rel_max = raw_relevances.min(), raw_relevances.max()
        if rel_max > rel_min:
            relevances = (raw_relevances - rel_min) / (rel_max - rel_min)
        else:
            relevances = np.ones_like(raw_relevances)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_indices) < final_k and remaining:
            best_idx = None
            best_score = -float("inf")

            for i in remaining:
                relevance = float(relevances[i])

                if not selected_indices:
                    redundancy = 0.0
                else:
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

    def _get_embeddings_for_chunks(self, chunks: list[RetrievedChunk]) -> list[np.ndarray]:
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
        index_path = self.index_dir / _INDEX_FILE
        meta_path = self.index_dir / _META_FILE

        faiss.write_index(self._index, str(index_path))

        meta = [chunk.model_dump() for chunk in self._chunks]
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        logger.info(f"Index saved → {index_path} ({self.size} vectors)")

    def _load(self) -> None:
        index_path = self.index_dir / _INDEX_FILE
        meta_path = self.index_dir / _META_FILE

        self._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text())
        self._chunks = [DocumentChunk(**m) for m in meta]

        logger.info(f"Index loaded ← {index_path} ({self.size} vectors)")

    def _persisted(self) -> bool:
        return (
            (self.index_dir / _INDEX_FILE).exists()
            and (self.index_dir / _META_FILE).exists()
        )

    def clear(self) -> None:
        self._chunks = []
        self._create_index()
        logger.info("Index cleared")

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self._chunks

    @property
    def documents(self) -> list[str]:
        return sorted({c.source_file for c in self._chunks})
