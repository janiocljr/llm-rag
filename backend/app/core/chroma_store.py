from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.models.schemas import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

_client: Optional[chromadb.HttpClient] = None
_pdf_col: Optional[chromadb.Collection] = None
_memory_col: Optional[chromadb.Collection] = None


def _get_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB connected → {settings.chroma_host}:{settings.chroma_port}")
    return _client


def get_pdf_collection() -> chromadb.Collection:
    global _pdf_col
    if _pdf_col is None:
        _pdf_col = _get_client().get_or_create_collection(
            name=settings.chroma_collection_embeddings,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{settings.chroma_collection_embeddings}' "
            f"— {_pdf_col.count()} vectors"
        )
    return _pdf_col


def get_memory_collection() -> chromadb.Collection:
    global _memory_col
    if _memory_col is None:
        _memory_col = _get_client().get_or_create_collection(
            name=settings.chroma_collection_memory,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{settings.chroma_collection_memory}' "
            f"— {_memory_col.count()} vectors"
        )
    return _memory_col


class ChromaPDFStore:
    def __init__(self) -> None:
        self._col = get_pdf_collection()

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> list[str]:
        if not chunks:
            return []

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "token_estimate": c.token_estimate,
                "citation": c.citation,
            }
            for c in chunks
        ]

        self._col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info(f"ChromaDB: upserted {len(chunks)} PDF chunks")
        return ids

    def clear(self) -> None:
        client = _get_client()
        client.delete_collection(settings.chroma_collection_embeddings)
        global _pdf_col
        _pdf_col = client.get_or_create_collection(
            name=settings.chroma_collection_embeddings,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB pdf_embeddings collection cleared")

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        threshold: float = 0.30,
        allowed_types: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        if self._col.count() == 0:
            logger.warning("ChromaDB pdf_embeddings is empty — run /ingest first")
            return []

        where: Optional[dict] = None

        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - dist
            if similarity < threshold:
                continue
            chunk = DocumentChunk(
                chunk_id=cid,
                text=doc,
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 0),
                chunk_index=meta.get("chunk_index", 0),
                total_chunks_in_page=0,
                char_count=len(doc),
                token_estimate=meta.get("token_estimate", 0),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=similarity))

        return retrieved

    @property
    def size(self) -> int:
        return self._col.count()

    @property
    def documents(self) -> list[str]:
        if self._col.count() == 0:
            return []
        result = self._col.get(include=["metadatas"])
        files: set[str] = set()
        for meta in result["metadatas"]:
            if meta.get("source_file"):
                files.add(meta["source_file"])
        return sorted(files)


class ChromaMemoryStore:
    def __init__(self) -> None:
        self._col = get_memory_collection()

    def save(self, text: str, embedding: list[float], memory_type: str, session_id: str, mongo_id: str = "", tags: Optional[list[str]] = None) -> str:
        memory_id = str(uuid.uuid4())
        self._col.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "memory_type": memory_type,
                "session_id": session_id,
                "mongo_id": mongo_id,
                "tags": ",".join(tags or []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        )
        logger.debug(f"Memory saved [{memory_type}] id={memory_id}")
        return memory_id

    def save_batch(self, memories: list[dict]) -> list[str]:
        if not memories:
            return []
        ids = [str(uuid.uuid4()) for _ in memories]
        embeddings = [m["embedding"] for m in memories]
        documents = [m["text"] for m in memories]
        metadatas = [
            {
                "memory_type": m.get("memory_type", "note"),
                "session_id": m.get("session_id", ""),
                "mongo_id": m.get("mongo_id", ""),
                "tags": ",".join(m.get("tags", [])),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for m in memories
        ]
        self._col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info(f"Memory batch saved: {len(memories)} units")
        return ids

    def recall(self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.50, session_id: Optional[str] = None, memory_types: Optional[list[str]] = None) -> list[dict]:
        if self._col.count() == 0:
            return []

        where_clauses: list[dict] = []
        if session_id:
            where_clauses.append({"session_id": {"$eq": session_id}})
        if memory_types and len(memory_types) == 1:
            where_clauses.append({"memory_type": {"$eq": memory_types[0]}})
        elif memory_types:
            where_clauses.append({"memory_type": {"$in": memory_types}})

        where: Optional[dict] = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        try:
            results = self._col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._col.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning(f"ChromaDB memory recall failed: {exc}")
            return []

        memories: list[dict] = []
        for cid, doc, meta, dist in zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = 1.0 - dist
            if similarity < threshold:
                continue
            memories.append({
                "chroma_id": cid,
                "text": doc,
                "memory_type": meta.get("memory_type", "note"),
                "session_id": meta.get("session_id", ""),
                "mongo_id": meta.get("mongo_id", ""),
                "tags": meta.get("tags", "").split(","),
                "timestamp": meta.get("timestamp", ""),
                "score": round(similarity, 4),
            })

        return memories

    @property
    def size(self) -> int:
        return self._col.count()
