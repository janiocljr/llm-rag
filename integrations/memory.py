from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.core.chroma_store import ChromaMemoryStore, ChromaPDFStore
from app.core.mongo_store import MongoDocumentStore, MongoSessionStore, MongoTaskStore
from app.core.config import settings

logger = logging.getLogger(__name__)

_pdf_store:      Optional[ChromaPDFStore]      = None
_memory_store:   Optional[ChromaMemoryStore]   = None
_doc_store:      Optional[MongoDocumentStore]  = None
_session_store:  Optional[MongoSessionStore]   = None
_task_store:     Optional[MongoTaskStore]      = None


def get_pdf_store() -> ChromaPDFStore:
    global _pdf_store
    if _pdf_store is None:
        _pdf_store = ChromaPDFStore()
    return _pdf_store


def get_memory_store() -> ChromaMemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = ChromaMemoryStore()
    return _memory_store


def get_doc_store() -> MongoDocumentStore:
    global _doc_store
    if _doc_store is None:
        _doc_store = MongoDocumentStore()
    return _doc_store


def get_session_store() -> MongoSessionStore:
    global _session_store
    if _session_store is None:
        _session_store = MongoSessionStore()
    return _session_store


def get_task_store() -> MongoTaskStore:
    global _task_store
    if _task_store is None:
        _task_store = MongoTaskStore()
    return _task_store


class MemoryOrchestrator:

    def __init__(self, embedder) -> None:
        self._embedder     = embedder
        self._pdf_store    = get_pdf_store()
        self._memory_store = get_memory_store()
        self._doc_store    = get_doc_store()
        self._session_store= get_session_store()
        self._task_store   = get_task_store()


    def new_session(self, title: str = "", tags: Optional[list[str]] = None) -> str:
        session_id = str(uuid.uuid4())
        self._session_store.create(session_id, title=title, tags=tags)
        logger.info(f"New session: {session_id}")
        return session_id

    def resume_session(self, session_id: str) -> Optional[dict]:
        session = self._session_store.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
        return session

    def close_session(self, session_id: str, summary: str = "") -> None:
        self._session_store.close(session_id, summary=summary)
        logger.info(f"Session closed: {session_id}")


    def reconstruct_context(
        self,
        question: str,
        session_id: str,
        top_k: int = 5,
        threshold: float = 0.50,
        exclude_current_session: bool = True,
    ) -> list[dict]:
        q_embedding = self._embed_text(question)

        memories = self._memory_store.recall(
            query_embedding=q_embedding,
            top_k=top_k * 2,
            threshold=threshold,
        )

        if exclude_current_session:
            memories = [m for m in memories if m.get("session_id") != session_id]

        return memories[:top_k]


    def save_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        tags: Optional[list[str]] = None,
    ) -> dict:
        q_embedding = self._embed_text(question)
        a_embedding = self._embed_text(answer)

        q_chroma_id = self._memory_store.save(
            text=question,
            embedding=q_embedding,
            memory_type="question",
            session_id=session_id,
            tags=tags or [],
        )
        a_chroma_id = self._memory_store.save(
            text=answer,
            embedding=a_embedding,
            memory_type="answer",
            session_id=session_id,
            tags=tags or [],
        )

        self._session_store.add_turn(session_id, "user", question, chroma_id=q_chroma_id)
        self._session_store.add_turn(session_id, "assistant", answer, chroma_id=a_chroma_id)

        content = f"## Pergunta\n\n{question}\n\n## Resposta\n\n{answer}"
        mongo_id = self._doc_store.create(
            title=question[:100],
            content=content,
            doc_type="conversation",
            tags=tags or [],
            session_id=session_id,
            chroma_ids=[q_chroma_id, a_chroma_id],
        )

        self._session_store.link_doc(session_id, mongo_id)

        return {
            "session_id":   session_id,
            "mongo_id":     mongo_id,
            "q_chroma_id":  q_chroma_id,
            "a_chroma_id":  a_chroma_id,
        }


    def save_note(
        self,
        title: str,
        content: str,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        embedding  = self._embed_text(content)
        chroma_id  = self._memory_store.save(
            text=content,
            embedding=embedding,
            memory_type="note",
            session_id=session_id or "",
            tags=tags or [],
        )
        mongo_id = self._doc_store.create(
            title=title,
            content=content,
            doc_type="note",
            tags=tags or [],
            session_id=session_id,
            chroma_ids=[chroma_id],
        )
        if session_id:
            self._session_store.link_doc(session_id, mongo_id)
        return {"mongo_id": mongo_id, "chroma_id": chroma_id}

    def save_knowledge(
        self,
        title: str,
        content: str,
        source_file: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        embedding = self._embed_text(content)
        chroma_id = self._memory_store.save(
            text=content,
            embedding=embedding,
            memory_type="note",
            session_id=session_id or "",
            tags=tags or [],
        )
        mongo_id = self._doc_store.create(
            title=title,
            content=content,
            doc_type="knowledge",
            tags=tags or [],
            source_file=source_file,
            session_id=session_id,
            chroma_ids=[chroma_id],
        )
        return {"mongo_id": mongo_id, "chroma_id": chroma_id}

    def save_task(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        content   = f"# Tarefa: {title}\n\n{description}"
        embedding = self._embed_text(content)
        chroma_id = self._memory_store.save(
            text=content,
            embedding=embedding,
            memory_type="task",
            session_id=session_id or "",
            tags=tags or [],
        )
        task_id = self._task_store.create(
            title=title,
            description=description,
            priority=priority,
            tags=tags or [],
            session_id=session_id,
            chroma_id=chroma_id,
        )
        mongo_id = self._doc_store.create(
            title=f"[TASK] {title}",
            content=content,
            doc_type="task",
            tags=tags or [],
            session_id=session_id,
            chroma_ids=[chroma_id],
            metadata={"task_id": task_id},
        )
        return {"task_id": task_id, "mongo_id": mongo_id, "chroma_id": chroma_id}


    def memory_stats(self) -> dict:
        return {
            "chroma": {
                "pdf_embeddings": self._pdf_store.size,
                "chat_memory":    self._memory_store.size,
            },
            "mongodb": self._doc_store.stats(),
        }


    def _embed_text(self, text: str) -> list[float]:
        vec = self._embedder.embed_query(text)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)
