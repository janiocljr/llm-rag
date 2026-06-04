"""
app/core/config.py  —  v3 (ChromaDB + MongoDB)
===============================================
Central configuration.  All values can be overridden via environment
variables or backend/.env.

New in v3
---------
CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION_EMBEDDINGS, CHROMA_COLLECTION_MEMORY
MONGO_URI, MONGO_DB
USE_CHROMA  — when True, ChromaDB is the vector store (FAISS is bypassed)
"""

from __future__ import annotations

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    data_dir:  Path = Path("data/pdfs")
    index_dir: Path = Path("data/index")
    model_dir: Path = Path("models")

    embedding_model:      str = "BAAI/bge-m3"
    embedding_dim:        int = 1024
    embedding_batch_size: int = 64

    chunk_size:    int = 512
    chunk_overlap: int = 64

    retrieval_top_k:      int   = 8
    retrieval_final_k:    int   = 3
    similarity_threshold: float = 0.30
    mmr_lambda:           float = 0.6

    use_bm25_hybrid:            bool = True
    auto_chunk_type_routing:    bool = True
    skip_table_chunks_in_index: bool = False
    min_chunk_tokens:           int  = 20

    use_chroma:                   bool = True
    chroma_host:                  str  = "localhost"
    chroma_port:                  int  = 8200
    chroma_collection_embeddings: str  = "pdf_embeddings"
    chroma_collection_memory:     str  = "chat_memory"

    mongo_uri: str = "mongodb://ragadmin:ragpassword@localhost:27017/rag_knowledge?authSource=admin"
    mongo_db:  str = "rag_knowledge"

    llm_model_path:     str   = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    llm_context_length: int   = 4096
    llm_max_new_tokens: int   = 512
    llm_temperature:    float = 0.1
    llm_n_gpu_layers:   int   = 0
    llm_n_threads:      int   = 8

    system_prompt: str = (
        "Você é um assistente especializado em análise de documentos institucionais "
        "e relatórios econômicos. Responda APENAS com base no contexto fornecido. "
        "Não use conhecimento externo. Se a informação não estiver no contexto, "
        "diga exatamente: 'Não encontrei essa informação nos documentos fornecidos.' "
        "Seja preciso, cite a fonte e a página para cada afirmação."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        return v

    @field_validator("mmr_lambda")
    @classmethod
    def validate_lambda(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        return v

    @field_validator("min_chunk_tokens")
    @classmethod
    def validate_min_tokens(cls, v: int) -> int:
        if v < 0:
            raise ValueError("min_chunk_tokens must be >= 0")
        return v


settings = Settings()
