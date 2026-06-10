from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        _client.admin.command("ping")
        logger.info(f"MongoDB connected → {settings.mongo_uri.split('@')[-1]}")
    return _client


def _db():
    return _get_client()[settings.mongo_db]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(doc: dict) -> dict:
    if "_id" in doc and isinstance(doc["_id"], ObjectId):
        doc["_id"] = str(doc["_id"])
    return doc


class MongoDocumentStore:
    @property
    def col(self) -> Collection:
        return _db()["documents"]

    def create(self, title: str, content: str, doc_type: str = "note", tags: Optional[list[str]] = None, source_file: Optional[str] = None, page_number: Optional[int] = None, session_id: Optional[str] = None, chroma_ids: Optional[list[str]] = None, status: str = "active", metadata: Optional[dict] = None) -> str:
        now = _now()
        doc = {
            "title": title,
            "content": content,
            "doc_type": doc_type,
            "tags": tags or [],
            "status": status,
            "source_file": source_file,
            "page_number": page_number,
            "session_id": session_id,
            "chroma_ids": chroma_ids or [],
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        }
        result = self.col.insert_one(doc)
        mongo_id = str(result.inserted_id)
        logger.debug(f"Document created [{doc_type}] id={mongo_id} title='{title[:60]}'")
        return mongo_id

    def create_from_pdf_chunk(self, chunk_text: str, source_file: str, page_number: int, chunk_id: str, chroma_ids: Optional[list[str]] = None, tags: Optional[list[str]] = None, session_id: Optional[str] = None) -> str:
        title = f"{source_file} — p.{page_number} [{chunk_id}]"
        return self.create(
            title=title,
            content=chunk_text,
            doc_type="pdf_chunk",
            tags=tags or [source_file],
            source_file=source_file,
            page_number=page_number,
            session_id=session_id,
            chroma_ids=chroma_ids or [chunk_id],
            metadata={"chunk_id": chunk_id},
        )

    def get_by_id(self, mongo_id: str) -> Optional[dict]:
        doc = self.col.find_one({"_id": ObjectId(mongo_id)})
        return _serialize(doc) if doc else None

    def get_by_chroma_id(self, chroma_id: str) -> Optional[dict]:
        doc = self.col.find_one({"chroma_ids": chroma_id})
        return _serialize(doc) if doc else None

    def search_text(self, query: str, doc_type: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 20) -> list[dict]:
        q: dict[str, Any] = {"$text": {"$search": query}}
        if doc_type:
            q["doc_type"] = doc_type
        if tags:
            q["tags"] = {"$in": tags}
        projection = {"score": {"$meta": "textScore"}}
        cursor = self.col.find(q, projection).sort([("score", {"$meta": "textScore"})]).limit(limit)
        return [_serialize(d) for d in cursor]

    def list_by_session(self, session_id: str) -> list[dict]:
        cursor = self.col.find({"session_id": session_id}).sort("created_at", ASCENDING)
        return [_serialize(d) for d in cursor]

    def list_recent(self, doc_type: Optional[str] = None, limit: int = 50) -> list[dict]:
        q = {}
        if doc_type:
            q["doc_type"] = doc_type
        cursor = self.col.find(q).sort("created_at", DESCENDING).limit(limit)
        return [_serialize(d) for d in cursor]

    def list_by_tags(self, tags: list[str], limit: int = 50) -> list[dict]:
        cursor = self.col.find({"tags": {"$in": tags}}).sort("created_at", DESCENDING).limit(limit)
        return [_serialize(d) for d in cursor]

    def update(self, mongo_id: str, **fields) -> bool:
        fields["updated_at"] = _now()
        result = self.col.update_one({"_id": ObjectId(mongo_id)}, {"$set": fields})
        return result.modified_count > 0

    def append_chroma_ids(self, mongo_id: str, chroma_ids: list[str]) -> bool:
        result = self.col.update_one({"_id": ObjectId(mongo_id)}, {"$addToSet": {"chroma_ids": {"$each": chroma_ids}}, "$set": {"updated_at": _now()}})
        return result.modified_count > 0

    def archive(self, mongo_id: str) -> bool:
        return self.update(mongo_id, status="archived")

    def delete(self, mongo_id: str) -> bool:
        result = self.col.delete_one({"_id": ObjectId(mongo_id)})
        return result.deleted_count > 0

    def stats(self) -> dict:
        pipeline = [{"$group": {"_id": "$doc_type", "count": {"$sum": 1}}}, {"$sort": {"count": DESCENDING}}]
        by_type = {r["_id"]: r["count"] for r in self.col.aggregate(pipeline)}
        return {"total": self.col.count_documents({}), "by_type": by_type}


class MongoSessionStore:
    @property
    def col(self) -> Collection:
        return _db()["sessions"]

    def create(self, session_id: str, title: str = "", tags: Optional[list[str]] = None) -> str:
        now = _now()
        doc = {
            "session_id": session_id,
            "title": title or f"Session {now.strftime('%Y-%m-%d %H:%M')}",
            "summary": "",
            "turns": [],
            "tags": tags or [],
            "started_at": now,
            "ended_at": None,
            "chroma_ids": [],
            "doc_ids": [],
            "metadata": {},
        }
        result = self.col.insert_one(doc)
        logger.info(f"Session created: {session_id}")
        return str(result.inserted_id)

    def add_turn(self, session_id: str, role: str, content: str, chroma_id: Optional[str] = None) -> bool:
        turn = {"role": role, "content": content, "timestamp": _now().isoformat(), "chroma_id": chroma_id}
        update: dict[str, Any] = {"$push": {"turns": turn}}
        if chroma_id:
            update["$addToSet"] = {"chroma_ids": chroma_id}
        result = self.col.update_one({"session_id": session_id}, update)
        return result.modified_count > 0

    def get(self, session_id: str) -> Optional[dict]:
        doc = self.col.find_one({"session_id": session_id})
        return _serialize(doc) if doc else None

    def list_recent(self, limit: int = 20) -> list[dict]:
        cursor = self.col.find({}).sort("started_at", DESCENDING).limit(limit)
        return [_serialize(d) for d in cursor]

    def close(self, session_id: str, summary: str = "") -> bool:
        result = self.col.update_one({"session_id": session_id}, {"$set": {"ended_at": _now(), "summary": summary}})
        return result.modified_count > 0

    def link_doc(self, session_id: str, mongo_doc_id: str) -> bool:
        result = self.col.update_one({"session_id": session_id}, {"$addToSet": {"doc_ids": mongo_doc_id}})
        return result.modified_count > 0


class MongoTaskStore:
    @property
    def col(self) -> Collection:
        return _db()["tasks"]

    def create(self, title: str, description: str = "", priority: str = "medium", tags: Optional[list[str]] = None, due_date: Optional[datetime] = None, session_id: Optional[str] = None, chroma_id: Optional[str] = None) -> str:
        now = _now()
        doc = {"title": title, "description": description, "status": "todo", "priority": priority, "tags": tags or [], "due_date": due_date, "session_id": session_id, "chroma_id": chroma_id, "created_at": now, "updated_at": now}
        result = self.col.insert_one(doc)
        return str(result.inserted_id)

    def update_status(self, task_id: str, status: str) -> bool:
        result = self.col.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status, "updated_at": _now()}})
        return result.modified_count > 0

    def list_open(self, priority: Optional[str] = None) -> list[dict]:
        q: dict = {"status": {"$in": ["todo", "in_progress"]}}
        if priority:
            q["priority"] = priority
        cursor = self.col.find(q).sort("priority", ASCENDING)
        return [_serialize(d) for d in cursor]

    def list_by_session(self, session_id: str) -> list[dict]:
        cursor = self.col.find({"session_id": session_id})
        return [_serialize(d) for d in cursor]
