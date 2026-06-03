"""
app/core/config.py
==================
Central configuration for the RAG system.

All values can be overridden via environment variables (or a .env file).
Pydantic-Settings handles type coercion and validation automatically.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single source of truth for every tuneable knob in the system.

    DESIGN RATIONALE
    ----------------
    Using Pydantic-Settings lets operators configure the system purely through
    environment variables (ideal for Docker / CI), while sane defaults make
    it "just work" out of the box for developers.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )




    data_dir: Path = Field(
        default=Path("data/pdfs"),
        description="Directory that contains the source PDF files.",
    )
    index_dir: Path = Field(
        default=Path("data/index"),
        description="Directory where FAISS index + metadata are persisted.",
    )
    model_dir: Path = Field(
        default=Path("models"),
        description="Directory where local LLM weights are stored.",
    )




    chunk_size: int = Field(
        default=512,
        description=(
            "Target number of TOKENS per chunk. "
            "512 is a sweet-spot: large enough to carry full sentences/paragraphs "
            "that give the LLM context, small enough that retrieved chunks are precise. "
            "Going above ~800 hurts retrieval precision; below ~256 loses semantic context."
        ),
    )
    chunk_overlap: int = Field(
        default=64,
        description=(
            "Token overlap between consecutive chunks. "
            "Overlap prevents answers from being split across chunk boundaries. "
            "~12 %% of chunk_size is the empirically-observed sweet-spot."
        ),
    )




    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description=(
            "HuggingFace model ID for sentence-transformers. "
            "bge-m3 é multilingual (100+ idiomas, incluindo PT-BR), 570 M params, 1024-dim. "
            "Alternativa leve: 'intfloat/multilingual-e5-small' (118 M, 384-dim)."
        ),
    )
    embedding_dim: int = Field(
        default=1024,
        description="Output dimension do modelo de embedding (deve corresponder ao modelo). bge-m3 = 1024.",
    )
    embedding_batch_size: int = Field(
        default=128,
        description="Batch size for embedding generation. Tune down on low-RAM machines.",
    )
    embedding_device: str = Field(
        default="auto",
        description=(
            "Device para embedding: 'auto' detecta GPU automaticamente "
            "(MPS no Apple Silicon, CUDA em NVIDIA). 'cpu' força CPU. 'mps' força Metal."
        ),
    )




    faiss_index_type: Literal["flat", "ivf", "hnsw"] = Field(
        default="flat",
        description=(
            "FAISS index type. "
            "'flat' = exact cosine search, best for < 100 k chunks (our use-case). "
            "'hnsw' = approximate, sub-linear search, better for > 500 k chunks. "
            "'ivf' = inverted-file, good middle-ground."
        ),
    )




    retrieval_top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve per query before re-ranking.",
    )
    retrieval_final_k: int = Field(
        default=3,
        description=(
            "Chunks actually injected into the LLM prompt after MMR de-duplication. "
            "Keeping this ≤ 3 avoids prompt bloat while retaining diversity."
        ),
    )
    similarity_threshold: float = Field(
        default=0.35,
        description=(
            "Minimum cosine similarity for a chunk to be considered relevant. "
            "Chunks below this score are discarded — the system replies 'not found' "
            "rather than hallucinating. Tune up to increase precision, down for recall."
        ),
    )
    mmr_lambda: float = Field(
        default=0.6,
        description=(
            "MMR (Maximal Marginal Relevance) trade-off: "
            "1.0 = pure relevance, 0.0 = pure diversity. "
            "0.6 balances both — avoids returning near-duplicate chunks."
        ),
    )




    llm_model_path: str = Field(
        default="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        description=(
            "Path to the GGUF model file for llama-cpp-python. "
            "Default: Mistral 7B Instruct Q4_K_M (~4.1 GB). "
            "Quantised to 4-bit so it runs on 8 GB RAM without a GPU."
        ),
    )
    llm_context_length: int = Field(
        default=4096,
        description="Context window size (tokens) for the LLM.",
    )
    llm_max_new_tokens: int = Field(
        default=512,
        description="Maximum tokens the LLM is allowed to generate per answer.",
    )
    llm_temperature: float = Field(
        default=0.1,
        description=(
            "Sampling temperature. Low (0.1) = deterministic/factual. "
            "Kept low to reduce hallucination risk in a RAG setting."
        ),
    )
    llm_n_gpu_layers: int = Field(
        default=0,
        description=(
            "Number of transformer layers to offload to GPU via llama.cpp. "
            "0 = CPU-only. Set to -1 to offload all layers if a GPU is available."
        ),
    )
    llm_n_threads: int = Field(
        default=8,
        description="CPU threads for llama.cpp inference. Match your machine's core count.",
    )




    system_prompt: str = Field(
        default=(
            "Você é um assistente especializado em análise de documentos. "
            "Responda SEMPRE em português brasileiro (pt-BR). "
            "Use APENAS as informações do contexto fornecido abaixo. "
            "Se o contexto não contiver informação suficiente para responder, diga exatamente: "
            "'Não encontrei essa informação nos documentos fornecidos.' "
            "NÃO use conhecimento externo. "
            "Sempre cite o documento de origem e número da página para cada informação."
        ),
    )




    use_chroma: bool = Field(default=False, description="Use ChromaDB instead of FAISS vector store")
    chroma_host: str = Field(default="localhost", description="ChromaDB HTTP host")
    chroma_port: int = Field(default=8200, description="ChromaDB HTTP port")
    chroma_collection_embeddings: str = Field(default="pdf_embeddings")
    chroma_collection_memory: str = Field(default="chat_memory")

    mongo_uri: str = Field(
        default="mongodb://ragadmin:ragpassword@localhost:27017/rag_knowledge?authSource=admin",
        description="MongoDB connection URI",
    )
    mongo_db: str = Field(default="rag_knowledge", description="MongoDB database name")




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



settings = Settings()
