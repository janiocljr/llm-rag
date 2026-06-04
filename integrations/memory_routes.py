"""
app/api/memory_routes.py
========================
REST endpoints for the persistent memory system.

Endpoints
---------
POST   /api/v1/memory/sessions                → new_session()
GET    /api/v1/memory/sessions                → list_recent()
GET    /api/v1/memory/sessions/{session_id}   → get session + turns
POST   /api/v1/memory/sessions/{session_id}/close  → close_session()

POST   /api/v1/memory/notes                   → save_note()
POST   /api/v1/memory/tasks                   → save_task()
POST   /api/v1/memory/knowledge               → save_knowledge()

GET    /api/v1/memory/documents               → list / search documents
GET    /api/v1/memory/stats                   → memory stats

POST   /api/v1/memory/recall                  → reconstruct context for a query
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


def _mem(request: Request):
    """Access MemoryOrchestrator from app.state."""
    return request.app.state.memory



class NewSessionRequest(BaseModel):
    title: str = ""
    tags:  list[str] = []


class NewSessionResponse(BaseModel):
    session_id: str


class CloseSessionRequest(BaseModel):
    summary: str = ""


class SaveNoteRequest(BaseModel):
    title:      str
    content:    str
    session_id: Optional[str] = None
    tags:       list[str]     = []


class SaveKnowledgeRequest(BaseModel):
    title:       str
    content:     str
    source_file: Optional[str] = None
    session_id:  Optional[str] = None
    tags:        list[str]     = []


class SaveTaskRequest(BaseModel):
    title:       str
    description: str        = ""
    priority:    str        = Field(default="medium", pattern="^(low|medium|high|critical)$")
    tags:        list[str]  = []
    session_id:  Optional[str] = None


class RecallRequest(BaseModel):
    question:   str
    session_id: str
    top_k:      int   = 5
    threshold:  float = 0.50



@router.post("/sessions", response_model=NewSessionResponse, summary="Start a new chat session")
async def new_session(body: NewSessionRequest, request: Request):
    """Create a new session record. Returns session_id for subsequent queries."""
    mem = _mem(request)
    session_id = mem.new_session(title=body.title, tags=body.tags)
    return NewSessionResponse(session_id=session_id)


@router.get("/sessions", summary="List recent sessions")
async def list_sessions(limit: int = Query(default=20, ge=1, le=100), request: Request = None):
    mem = _mem(request)
    sessions = mem._session_store.list_recent(limit=limit)
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/sessions/{session_id}", summary="Get a session with its turn history")
async def get_session(session_id: str, request: Request):
    mem = _mem(request)
    session = mem.resume_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session


@router.post("/sessions/{session_id}/close", summary="Close a session with an optional summary")
async def close_session(session_id: str, body: CloseSessionRequest, request: Request):
    mem = _mem(request)
    mem.close_session(session_id, summary=body.summary)
    return {"ok": True, "session_id": session_id}



@router.post("/notes", summary="Save a markdown note to the knowledge base")
async def save_note(body: SaveNoteRequest, request: Request):
    mem = _mem(request)
    result = mem.save_note(
        title=body.title,
        content=body.content,
        session_id=body.session_id,
        tags=body.tags,
    )
    return {"ok": True, **result}


@router.post("/knowledge", summary="Save a derived knowledge fragment")
async def save_knowledge(body: SaveKnowledgeRequest, request: Request):
    mem = _mem(request)
    result = mem.save_knowledge(
        title=body.title,
        content=body.content,
        source_file=body.source_file,
        session_id=body.session_id,
        tags=body.tags,
    )
    return {"ok": True, **result}


@router.post("/tasks", summary="Save a task to the knowledge base")
async def save_task(body: SaveTaskRequest, request: Request):
    mem = _mem(request)
    result = mem.save_task(
        title=body.title,
        description=body.description,
        priority=body.priority,
        tags=body.tags,
        session_id=body.session_id,
    )
    return {"ok": True, **result}



@router.get("/documents", summary="List or search knowledge base documents")
async def list_documents(
    request: Request,
    doc_type: Optional[str] = Query(default=None),
    tags:     Optional[str] = Query(default=None, description="Comma-separated tags"),
    q:        Optional[str] = Query(default=None, description="Full-text search query"),
    limit:    int            = Query(default=30, ge=1, le=200),
):
    mem = _mem(request)
    doc_store = mem._doc_store

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    if q:
        docs = doc_store.search_text(q, doc_type=doc_type, tags=tag_list, limit=limit)
    elif tag_list:
        docs = doc_store.list_by_tags(tag_list, limit=limit)
    else:
        docs = doc_store.list_recent(doc_type=doc_type, limit=limit)

    return {"documents": docs, "total": len(docs)}


@router.get("/documents/{mongo_id}", summary="Get a single knowledge document by ID")
async def get_document(mongo_id: str, request: Request):
    mem = _mem(request)
    doc = mem._doc_store.get_by_id(mongo_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {mongo_id}")
    return doc



@router.post("/recall", summary="Recall semantically relevant past memories for a query")
async def recall_context(body: RecallRequest, request: Request):
    """
    Surface past memories most relevant to the given question.
    Used by the frontend to show the user what context will be injected.
    """
    mem = _mem(request)
    memories = mem.reconstruct_context(
        question=body.question,
        session_id=body.session_id,
        top_k=body.top_k,
        threshold=body.threshold,
    )
    return {"memories": memories, "total": len(memories)}



@router.get("/stats", summary="Memory system stats (ChromaDB + MongoDB)")
async def memory_stats(request: Request):
    mem = _mem(request)
    return mem.memory_stats()
