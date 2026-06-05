"""
app/core/pipeline.py  —  v3 (ChromaDB + MongoDB)
==================================================
RAG Pipeline — orchestrates all components end-to-end.

New in v3
---------
- VectorStore is now ChromaPDFStore when USE_CHROMA=True (default).
- MemoryOrchestrator is initialised at startup and exposed on app.state.
- query() accepts session_id; past memories are injected into the prompt.
- save_turn() is called automatically after each successful query.
- Fallback to FAISS VectorStore when USE_CHROMA=False.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from app.core.config import Settings
from app.core.embedder import Embedder
from app.core.ingestion import PDFIngester
from app.core.llm import LocalLLM, build_prompt, NOT_FOUND_SENTINEL
from app.core.memory import MemoryOrchestrator, get_pdf_store
from app.models.schemas import (
    DocumentChunk,
    IngestResponse,
    IndexStatsResponse,
    QueryResponse,
    QueryRequest,
    RetrievedChunk,
    RetrievedChunkResponse,
)

logger = logging.getLogger(__name__)

_STAT_QUERY_PATTERN = re.compile(
    r"\b(percentual|percentagem|porcento|quanto|quantos|quantas|"
    r"valor|valores|total|soma|média|taxa|índice|indicador|"
    r"exporta[çc][aã]o|importa[çc][aã]o|pib|gdp|milh[oõ]|bilh[oõ])\b",
    re.IGNORECASE,
)


def _is_statistical_query(question: str) -> bool:
    return bool(_STAT_QUERY_PATTERN.search(question))


def _clean_query(question: str) -> str:
    fillers = [
        r"^(me\s+fale\s+(sobre|a\s+respeito\s+de)|fale\s+sobre)\s+",
        r"^(o\s+que\s+é|qual\s+(é|foi|são|foram))\s+",
        r"^(pode\s+me\s+dizer|você\s+sabe)\s+",
    ]
    q = question.strip()
    for pattern in fillers:
        q = re.sub(pattern, "", q, flags=re.IGNORECASE)
    return q.strip() or question


class RAGPipeline:
    """
    Full RAG pipeline with persistent memory.

    app.state exposes:
        pipeline        → this object
        memory          → MemoryOrchestrator (used by memory API routes)
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        settings.model_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = Embedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )

        if settings.use_chroma:
            self.vector_store = get_pdf_store()
            logger.info("Vector store: ChromaDB")
        else:
            from app.core.vector_store import VectorStore
            self.vector_store = VectorStore(
                embedding_dim=settings.embedding_dim,
                index_dir=settings.index_dir,
                use_bm25=settings.use_bm25_hybrid,
            )
            logger.info("Vector store: FAISS (local)")

        self.memory = MemoryOrchestrator(embedder=self.embedder)

        self.llm = LocalLLM(
            model_path=settings.llm_model_path,
            context_length=settings.llm_context_length,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
            n_gpu_layers=settings.llm_n_gpu_layers,
            n_threads=settings.llm_n_threads,
        )

        self.ingester = PDFIngester(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )


    def ingest(self, force_reindex: bool = False) -> IngestResponse:
        t0 = time.perf_counter()

        if force_reindex:
            logger.info("Force re-index: clearing vector store")
            self.vector_store.clear()

        if self.vector_store.size > 0 and not force_reindex:
            logger.info(
                f"Index already has {self.vector_store.size} chunks — "
                "skipping. Use force_reindex=True to rebuild."
            )
            return IngestResponse(
                chunks_indexed=self.vector_store.size,
                documents_processed=len(self.vector_store.documents),
                latency_ms=0.0,
            )

        logger.info(f"Loading PDFs from {self.settings.data_dir}")
        chunks = self.ingester.load_directory(self.settings.data_dir)

        if self.settings.skip_table_chunks_in_index:
            before = len(chunks)
            chunks = [c for c in chunks if c.chunk_type != "table"]
            logger.info(f"Skipped {before - len(chunks)} table chunks")

        min_tokens = self.settings.min_chunk_tokens
        before = len(chunks)
        chunks = [c for c in chunks if c.token_estimate >= min_tokens]
        if len(chunks) < before:
            logger.info(f"Removed {before - len(chunks)} short chunks")

        logger.info(f"Embedding {len(chunks)} chunks…")
        embeddings = self.embedder.embed_documents(chunks)

        if self.settings.use_chroma:
            emb_list = embeddings.tolist()
            chroma_ids = self.vector_store.add(chunks, emb_list)

            doc_store = self.memory._doc_store
            for chunk, cid in zip(chunks, chroma_ids):
                doc_store.create_from_pdf_chunk(
                    chunk_text=chunk.text,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    chroma_ids=[cid],
                    tags=[chunk.source_file, chunk.chunk_type],
                )
        else:
            self.vector_store.add(chunks, embeddings)
            self.vector_store.save()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Ingestion complete: {len(chunks)} chunks from "
            f"{len(self.vector_store.documents)} documents in {elapsed_ms:.0f} ms"
        )

        return IngestResponse(
            chunks_indexed=len(chunks),
            documents_processed=len(self.vector_store.documents),
            latency_ms=elapsed_ms,
        )


    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Full RAG pipeline with persistent memory.

        Steps:
        1.  Clean + embed the question.
        2.  Recall relevant past memories from ChromaDB chat_memory.
        3.  Retrieve relevant PDF chunks (ChromaDB or FAISS).
        4.  MMR re-rank for diversity.
        5.  Build prompt = system + past memories + PDF context + question.
        6.  Generate LLM response.
        7.  Save Q&A turn to ChromaDB + MongoDB.
        """
        t0 = time.perf_counter()

        session_id = getattr(request, "session_id", None) or ""
        cleaned_q  = _clean_query(request.question)
        top_k      = request.top_k or self.settings.retrieval_top_k
        threshold  = request.similarity_threshold or self.settings.similarity_threshold

        t_embed = time.perf_counter()
        q_emb_np = self.embedder.embed_query(cleaned_q)
        q_emb    = q_emb_np.tolist() if hasattr(q_emb_np, "tolist") else list(q_emb_np)
        logger.debug(f"Embed: {(time.perf_counter()-t_embed)*1000:.1f}ms")

        past_memories: list[dict] = []
        if session_id:
            past_memories = self.memory.reconstruct_context(
                question=cleaned_q,
                session_id=session_id,
                top_k=3,
                threshold=0.50,
            )
            logger.debug(f"Recalled {len(past_memories)} past memories")

        if self.settings.auto_chunk_type_routing:
            allowed_types = None if _is_statistical_query(request.question) else ["text"]
        else:
            allowed_types = None

        t_ret = time.perf_counter()
        if self.settings.use_chroma:
            candidates = self.vector_store.search(
                query_embedding=q_emb,
                top_k=top_k,
                threshold=threshold,
                allowed_types=allowed_types,
            )
        else:
            if self.settings.use_bm25_hybrid and self.vector_store.use_bm25:
                candidates = self.vector_store.hybrid_search(
                    query=cleaned_q,
                    query_embedding=q_emb_np,
                    top_k=top_k,
                    threshold=threshold,
                    allowed_types=allowed_types,
                )
            else:
                candidates = self.vector_store.search(
                    query_embedding=q_emb_np,
                    top_k=top_k,
                    threshold=threshold,
                    allowed_types=allowed_types,
                )

        logger.info(
            f"Retrieval: {len(candidates)} candidates in {(time.perf_counter()-t_ret)*1000:.1f}ms"
        )

        if self.settings.use_chroma:
            final_chunks = candidates[: self.settings.retrieval_final_k]
        else:
            final_chunks = self.vector_store.mmr_rerank(
                candidates=candidates,
                query_embedding=q_emb_np,
                final_k=self.settings.retrieval_final_k,
                lambda_=self.settings.mmr_lambda,
            )

        found = len(final_chunks) > 0

        memory_context = _format_memory_context(past_memories)
        prompt = build_prompt(
            question=request.question,
            retrieved_chunks=final_chunks,
            system_prompt=self.settings.system_prompt,
            extra_context=memory_context,
        )

        t_llm = time.perf_counter()
        if found:
            raw_answer = self.llm.generate(prompt)
        else:
            raw_answer = NOT_FOUND_SENTINEL
        logger.debug(f"LLM: {(time.perf_counter()-t_llm)*1000:.1f}ms")

        answer_is_found = found and NOT_FOUND_SENTINEL not in raw_answer
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if session_id and answer_is_found:
            try:
                self.memory.save_turn(
                    session_id=session_id,
                    question=request.question,
                    answer=raw_answer,
                )
            except Exception as exc:
                logger.warning(f"Memory save failed (non-fatal): {exc}")

        return QueryResponse(
            question=request.question,
            answer=raw_answer,
            retrieved_chunks=[
                RetrievedChunkResponse(
                    chunk_id=rc.chunk.chunk_id,
                    source_file=rc.chunk.source_file,
                    page_number=rc.chunk.page_number,
                    chunk_index=rc.chunk.chunk_index,
                    score=rc.score,
                    text=rc.chunk.text,
                    citation=rc.chunk.citation,
                    chunk_type=rc.chunk.chunk_type,
                )
                for rc in final_chunks
            ],
            full_prompt=prompt,
            found_in_documents=answer_is_found,
            latency_ms=elapsed_ms,
        )


    def get_stats(self) -> IndexStatsResponse:
        index_type = (
            "ChromaDB (HNSW cosine)"
            if self.settings.use_chroma
            else "FAISS FlatIP + BM25"
        )
        return IndexStatsResponse(
            total_chunks=self.vector_store.size,
            documents=self.vector_store.documents,
            index_type=index_type,
            embedding_model=self.settings.embedding_model,
            embedding_dim=self.settings.embedding_dim,
        )


def _format_memory_context(memories: list[dict]) -> str:
    """
    Format recalled memories as a context block for the LLM prompt.
    Returns empty string when there are no memories.
    """
    if not memories:
        return ""
    lines = ["<memória_de_sessões_anteriores>"]
    for m in memories:
        ts = m.get("timestamp", "")[:19]
        mtype = m.get("memory_type", "note")
        lines.append(f"[{ts}] ({mtype}) {m['text'][:300]}")
    lines.append("</memória_de_sessões_anteriores>")
    return "\n".join(lines)
