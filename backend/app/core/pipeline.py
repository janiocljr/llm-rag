import json
import logging
import time
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.embedder import Embedder
from app.core.ingestion import PDFIngester
from app.core.lexical import BM25Index
from app.core.llm import (
    LocalLLM,
    NOT_FOUND_SENTINEL,
    build_messages,
    classify_answer,
    context_char_budget,
)
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

_SIGNATURE_FILE = "index_signature.json"


class RAGPipeline:
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

        self.lexical_index = BM25Index()
        self._chunk_pos: dict[str, int] = {}
        self._rebuild_lexical_index()

        self._check_index_signature()

    def _rebuild_lexical_index(self) -> None:
        chunks = self.vector_store.chunks
        self.lexical_index.build([c.text for c in chunks])
        self._chunk_pos = {c.chunk_id: i for i, c in enumerate(chunks)}
        if chunks:
            logger.info(f"BM25 lexical index built over {len(chunks)} chunks")

    def _index_signature(self) -> dict:
        return {
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
            "query_prefix": self.embedder.query_prefix,
            "document_prefix": self.embedder.document_prefix,
        }

    def _save_index_signature(self) -> None:
        path = self.settings.index_dir / _SIGNATURE_FILE
        path.write_text(json.dumps(self._index_signature(), ensure_ascii=False, indent=2))

    def _check_index_signature(self) -> None:
        """Detecta índice construído com outro modelo/protocolo de embedding.

        Buscar num índice cujos vetores foram gerados com modelo ou prefixos
        diferentes dos atuais produz scores sem significado — sintoma típico:
        nenhum resultado relevante. Nesse caso o índice precisa ser
        reconstruído via POST /api/v1/ingest {"force_reindex": true}.
        """
        if self.vector_store.size == 0:
            return

        path = self.settings.index_dir / _SIGNATURE_FILE
        current = self._index_signature()
        if not path.exists():
            logger.warning(
                "Índice existente sem assinatura de embedding (%d chunks). "
                "Ele pode ter sido construído com protocolo antigo — "
                "recomendado reconstruir: POST /api/v1/ingest {\"force_reindex\": true}. "
                "Assinatura atual: %s",
                self.vector_store.size,
                current,
            )
            return

        try:
            saved = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Assinatura do índice ilegível em %s", path)
            return

        if saved != current:
            logger.warning(
                "INCOMPATIBILIDADE de embedding: índice construído com %s, "
                "mas a configuração atual é %s. As buscas retornarão resultados "
                "ruins ou vazios até reconstruir o índice: "
                "POST /api/v1/ingest {\"force_reindex\": true}.",
                saved,
                current,
            )

    def ingest(self, force_reindex: bool = False) -> IngestResponse:
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
        self._save_index_signature()
        self._rebuild_lexical_index()

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

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, list[RetrievedChunk], list[RetrievedChunk]]:
        """Busca híbrida: vetorial (FAISS) + lexical (BM25), fusão RRF, MMR.

        A busca vetorial captura semântica; a lexical captura termos exatos
        (números, datas, siglas) que embeddings pequenos não distinguem.
        Retorna (query_embedding, candidatos_fundidos, chunks_finais).
        """
        top_k = top_k or self.settings.retrieval_top_k
        threshold = (
            threshold
            if threshold is not None
            else self.settings.similarity_threshold
        )

        t_start = time.perf_counter()
        query_embedding = self.embedder.embed_query(question)
        t_embed = time.perf_counter()

        vector_hits = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )
        lexical_hits = self.lexical_index.search(question, top_k=top_k)

        # Reciprocal Rank Fusion: robusta a escalas distintas de score.
        rrf_k = 60
        fused: dict[int, float] = {}
        for rank, rc in enumerate(vector_hits):
            pos = self._chunk_pos[rc.chunk.chunk_id]
            fused[pos] = fused.get(pos, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (pos, _) in enumerate(lexical_hits):
            fused[pos] = fused.get(pos, 0.0) + 1.0 / (rrf_k + rank + 1)

        t_retr = time.perf_counter()
        logger.info(
            "Embedding: %.3fs — vetorial: %d hits, lexical: %d hits, fundidos: %d",
            t_embed - t_start, len(vector_hits), len(lexical_hits), len(fused),
        )

        if not fused:
            return query_embedding, [], []

        chunks = self.vector_store.chunks
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        candidates = [
            RetrievedChunk(
                chunk=chunks[pos],
                score=self.vector_store.score_at(pos, query_embedding),
            )
            for pos, _ in ranked
        ]

        final_chunks = self.vector_store.mmr_rerank(
            candidates=candidates,
            query_embedding=query_embedding,
            final_k=self.settings.retrieval_final_k,
            lambda_=self.settings.mmr_lambda,
            relevance_scores=[score for _, score in ranked],
        )
        logger.info(
            "MMR re-rank: %.3fs — final chunks: %d",
            time.perf_counter() - t_retr, len(final_chunks),
        )
        return query_embedding, candidates, final_chunks

    def query(self, request: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()

        _, _, final_chunks = self.retrieve(
            question=request.question,
            top_k=request.top_k,
            threshold=request.similarity_threshold,
        )
        t_mmr = time.perf_counter()

        found = len(final_chunks) > 0

        messages = build_messages(
            question=request.question,
            retrieved_chunks=final_chunks,
            system_prompt=self.settings.system_prompt,
            max_context_chars=context_char_budget(
                self.settings.llm_context_length,
                self.settings.llm_max_new_tokens,
            ),
        )
        prompt_str = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        t_prompt = time.perf_counter()
        logger.info("Prompt build time: %.3fs — prompt length: %d chars",
                    t_prompt - t_mmr, len(prompt_str))

        if found:
            gen_start = time.perf_counter()
            raw_answer = self.llm.generate(messages)
            gen_end = time.perf_counter()
            logger.info("LLM generation time: %.3fs", gen_end - gen_start)
            raw_answer, llm_found = classify_answer(raw_answer)
        else:
            raw_answer = NOT_FOUND_SENTINEL
            llm_found = False

        answer_is_found = found and llm_found

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
            full_prompt=prompt_str,
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
