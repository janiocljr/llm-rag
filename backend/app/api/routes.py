import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import json

from app.models.schemas import (
    IndexStatsResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _pipeline(request: Request):
    return request.app.state.pipeline


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    try:
        return _pipeline(request).ingest(force_reindex=body.force_reindex)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    if _pipeline(request).vector_store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Call POST /ingest first.",
        )

    try:
        return _pipeline(request).query(body)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query/stream", tags=["query"])
async def query_stream(body: QueryRequest, request: Request):
    if _pipeline(request).vector_store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Call POST /ingest first.",
        )

    pipeline = _pipeline(request)

    try:
        _, _, final_chunks = pipeline.retrieve(
            question=body.question,
            top_k=body.top_k,
            threshold=body.similarity_threshold,
        )

        from app.core.llm import (
            build_messages,
            context_char_budget,
            stream_with_false_negative_guard,
        )

        full_prompt = build_messages(
            question=body.question,
            retrieved_chunks=final_chunks,
            system_prompt=pipeline.settings.system_prompt,
            max_context_chars=context_char_budget(
                pipeline.settings.llm_context_length,
                pipeline.settings.llm_max_new_tokens,
            ),
        )

        def event_stream():
            meta = {
                "type": "meta",
                "retrieved_chunks": [
                    {
                        "chunk_id": rc.chunk.chunk_id,
                        "source_file": rc.chunk.source_file,
                        "page_number": rc.chunk.page_number,
                        "chunk_index": rc.chunk.chunk_index,
                        "score": rc.score,
                        "text": rc.chunk.text,
                    }
                    for rc in final_chunks
                ],
                "full_prompt": full_prompt,
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            if final_chunks:
                guarded = stream_with_false_negative_guard(
                    pipeline.llm.stream_generate(full_prompt)
                )
                for token in guarded:
                    payload = {"type": "token", "text": token}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            else:
                payload = {"type": "complete", "text": "Não encontrei essa informação nos documentos fornecidos."}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type':'done'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as exc:
        logger.exception("Streaming query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats", response_model=IndexStatsResponse, tags=["system"])
async def stats(request: Request) -> IndexStatsResponse:
    return _pipeline(request).get_stats()
