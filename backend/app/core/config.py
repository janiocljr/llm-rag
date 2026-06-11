from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data/pdfs"))
    index_dir: Path = Field(default=Path("data/index"))
    model_dir: Path = Field(default=Path("models"))

    chunk_size: int = Field(default=300)
    chunk_overlap: int = Field(default=50)

    embedding_model: str = Field(default="intfloat/multilingual-e5-small")
    embedding_dim: int = Field(default=384)
    embedding_batch_size: int = Field(default=128)
    embedding_device: str = Field(default="auto")

    faiss_index_type: Literal["flat", "ivf", "hnsw"] = Field(default="flat")

    retrieval_top_k: int = Field(default=20)
    retrieval_final_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.50)
    mmr_lambda: float = Field(default=0.75)

    llm_model_path: str = Field(default="models/qwen2.5-7b-instruct-q4_k_m.gguf")
    llm_context_length: int = Field(default=4096)
    llm_max_new_tokens: int = Field(default=512)
    llm_temperature: float = Field(default=0.1)
    llm_n_gpu_layers: int = Field(default=0)
    llm_n_threads: int = Field(default=8)

    system_prompt: str = Field(
        default=(
            "Você é um assistente especializado em análise de documentos. "
            "Responda SEMPRE em português brasileiro (pt-BR). "
            "Use APENAS as informações do contexto fornecido — NÃO use conhecimento externo. "
            "REGRA PRINCIPAL: se o contexto contiver QUALQUER dado relacionado ao tema da "
            "pergunta, apresente esse dado na resposta, citando documento e página — mesmo "
            "que o período, recorte ou escopo seja um pouco diferente do perguntado. Nesse "
            "caso, apresente o dado disponível e explicite a diferença (ex.: pergunta sobre "
            "julho, dado disponível de junho → informe o valor de junho dizendo que se refere "
            "a junho). A frase 'Não encontrei essa informação nos documentos fornecidos.' é "
            "reservada EXCLUSIVAMENTE para quando nada no contexto tem relação com o tema da "
            "pergunta — usá-la quando existe dado relacionado é um erro grave. "
            "Sempre cite o documento de origem e número da página para cada informação."
        ),
    )

    use_chroma: bool = Field(default=False)
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8200)
    chroma_collection_embeddings: str = Field(default="pdf_embeddings")
    chroma_collection_memory: str = Field(default="chat_memory")

    mongo_uri: str = Field(
        default="mongodb://ragadmin:ragpassword@localhost:27017/rag_knowledge?authSource=admin"
    )
    mongo_db: str = Field(default="rag_knowledge")

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
