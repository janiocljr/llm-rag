"""
app/core/pipeline.py
====================
RAG Pipeline — orchestrates all components end-to-end.

This is the central coordinator. It wires together:
  Ingestion → Embedding → VectorStore → Retrieval → MMR → Prompt → LLM

Design principles:
- Each component is independently testable (see tests/).
- The pipeline is stateless per-query: the same instance handles concurrent requests.
- Hallucination control is enforced at two layers:
    Layer 1 (retrieval): similarity threshold filters out low-confidence chunks.
    Layer 2 (prompt): LLM is explicitly instructed to use ONLY context.
"""

import logging
import time
from pathlib import Path

from app.core.config import Settings
from app.core.embedder import Embedder
from app.core.ingestion import PDFIngester
from app.core.llm import LocalLLM, build_prompt, NOT_FOUND_SENTINEL
from app.core.vector_store import VectorStore
from app.core.memory import get_pdf_store, get_doc_store, get_session_store
import uuid
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


class RAGPipeline:
    """
    Full RAG pipeline: ingest PDFs → index → query → respond.

    Lifecycle:
    1. __init__: load embedder + LLM into memory, restore index if it exists.
    2. ingest():  parse PDFs, embed chunks, populate FAISS index.
    3. query():   embed question, retrieve chunks, re-rank, build prompt, generate.
    """

    def __init__(self, settings: Settings):
        self.settings = settings


        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        settings.model_dir.mkdir(parents=True, exist_ok=True)


        self.embedder = Embedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )


        self.vector_store = VectorStore(
            embedding_dim=settings.embedding_dim,
            index_dir=settings.index_dir,
        )


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
            index_dir=settings.index_dir,
        )





    def ingest(self, force_reindex: bool = False) -> IngestResponse:
        """
        Parse all PDFs in data_dir, embed chunks, and index them.

        force_reindex=True: wipe the existing index first.
        force_reindex=False: skip if the index already has data.
        """
        t0 = time.perf_counter()

        if force_reindex:
            logger.info("Force re-index: clearing existing index")
            self.vector_store.clear()

        if self.vector_store.size > 0 and not force_reindex:
            logger.info(
                f"Index already contains {self.vector_store.size} chunks — "
                "skipping ingestion. Use force_reindex=True to rebuild."
            )
            return IngestResponse(
                chunks_indexed=self.vector_store.size,
                documents_processed=len(self.vector_store.documents),
                latency_ms=0.0,
            )


        logger.info(f"Loading PDFs from {self.settings.data_dir}")
        chunks = self.ingester.load_directory(self.settings.data_dir)


        logger.info(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.embed_documents(chunks)


        self.vector_store.add(chunks, embeddings)


        self.vector_store.save()




        try:
            pdf_store = get_pdf_store()
            doc_store = get_doc_store()

            try:

                emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
                chroma_ids = pdf_store.add(chunks=chunks, embeddings=emb_list)
                logger.info(f"ChromaDB: upserted {len(chroma_ids)} chunks")




                try:
                    session_store = get_session_store()
                    session_id = str(uuid.uuid4())

                    session_store.create(session_id, title=f"ingest-{session_id}", tags=["ingest"])
                except Exception:
                    session_id = ""


                for chunk, cid in zip(chunks, chroma_ids):
                    try:
                        doc_store.create_from_pdf_chunk(
                            chunk_text=chunk.text,
                            source_file=chunk.source_file,
                            page_number=chunk.page_number,
                            chunk_id=chunk.chunk_id,
                            chroma_ids=[cid],
                            session_id=session_id,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to create Mongo document for chunk {chunk.chunk_id}: {exc}")

            except Exception as exc:
                logger.warning(f"ChromaDB upsert failed: {exc}")
        except Exception:
            logger.info("ChromaDB/MongoDB not available — skipping persistent memory sync")

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
        Full RAG query pipeline.

        Steps:
        1. Embed the question.
        2. Retrieve top-k similar chunks (above threshold).
        3. MMR re-rank to remove near-duplicates.
        4. Build structured prompt.
        5. Generate LLM response.
        6. Detect "not found" sentinel.
        7. Return full response including prompt and retrieved chunks.
        """
        t0 = time.perf_counter()

        top_k = request.top_k or self.settings.retrieval_top_k
        threshold = request.similarity_threshold or self.settings.similarity_threshold


        t_start = time.perf_counter()
        query_embedding = self.embedder.embed_query(request.question)
        t_embed = time.perf_counter()


        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )
        t_retr = time.perf_counter()
        logger.info("Embedding time: %.3fs — retrieved %d candidates",
                    t_embed - t_start, len(candidates))
        logger.info("Candidate scores: %s",
                    [round(c.score, 4) for c in candidates])


        final_chunks = self.vector_store.mmr_rerank(
            candidates=candidates,
            query_embedding=query_embedding,
            final_k=self.settings.retrieval_final_k,
            lambda_=self.settings.mmr_lambda,
        )
        t_mmr = time.perf_counter()
        logger.info("MMR re-rank time: %.3fs — final chunks: %d",
                    t_mmr - t_retr, len(final_chunks))

        found = len(final_chunks) > 0


        prompt = build_prompt(
            question=request.question,
            retrieved_chunks=final_chunks,
            system_prompt=self.settings.system_prompt,
        )
        t_prompt = time.perf_counter()
        logger.info("Prompt build time: %.3fs — prompt length: %d chars",
                    t_prompt - t_mmr, len(prompt))


        if found:
            gen_start = time.perf_counter()
            raw_answer = self.llm.generate(prompt)
            gen_end = time.perf_counter()
            logger.info("LLM generation time: %.3fs", gen_end - gen_start)
        else:


            raw_answer = NOT_FOUND_SENTINEL


        answer_is_found = found and NOT_FOUND_SENTINEL not in raw_answer


        elapsed_ms = (time.perf_counter() - t0) * 1000

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
                )
                for rc in final_chunks
            ],
            full_prompt=prompt,
            found_in_documents=answer_is_found,
            latency_ms=elapsed_ms,
        )
    def get_stats(self) -> IndexStatsResponse:
        return IndexStatsResponse(
            total_chunks=self.vector_store.size,
            documents=self.vector_store.documents,
            index_type="FlatIP (exact cosine)",
            embedding_model=self.settings.embedding_model,
            embedding_dim=self.settings.embedding_dim,
        )
