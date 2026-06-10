import logging
from pathlib import Path

from llama_cpp import Llama

from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

NOT_FOUND_SENTINEL = "Não encontrei essa informação nos documentos fornecidos."


class LocalLLM:
    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        n_gpu_layers: int = 0,
        n_threads: int = 8,
    ):
        model_path_ = Path(model_path)
        if not model_path_.exists():
            logger.warning(
                f"LLM model not found at {model_path_}. "
                "Running in STUB MODE — responses will be placeholder text."
            )
            self._stub = True
            self._model = None
        else:
            logger.info(f"Loading LLM: {model_path_} ...")
            self._stub = False
            self._model = Llama(
                model_path=str(model_path_),
                n_ctx=context_length,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            logger.info("LLM loaded successfully")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        if self._stub:
            return (
                "[STUB MODE] LLM model not loaded. "
                f"Would have generated a response for {len(messages)} messages."
            )

        output = self._model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return output["choices"][0]["message"]["content"].strip()

    def stream_generate(self, messages: list[dict]):
        if self._stub:
            yield "[STUB MODE] LLM model not loaded."
            return

        for chunk in self._model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                yield text


def build_messages(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    system_prompt: str,
) -> list[dict]:
    if retrieved_chunks:
        context_parts = []
        for i, rc in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Fonte {i}: {rc.chunk.source_file} | Página {rc.chunk.page_number} | "
                f"Relevância: {rc.score:.1%}]\n{rc.chunk.text}"
            )
        context_block = "\n\n---\n\n".join(context_parts)
        context_intro = f"Você tem {len(retrieved_chunks)} trechos relevantes dos documentos para responder:"
    else:
        context_block = ""
        context_intro = "(Nenhum contexto relevante encontrado nos documentos.)"

    user_content = (
        f"{context_intro}\n\n"
        f"<context>\n{context_block}\n</context>\n\n"
        f"Pergunta: {question}\n\n"
        f"Instruções:\n"
        f"1. Use TODOS os trechos acima para construir uma resposta completa.\n"
        f"2. Integre informações de múltiplos trechos se relevante.\n"
        f"3. Para cada informação, cite a fonte como [Fonte N, Página X].\n"
        f"4. Se os trechos não contiverem informação para responder, responda: "
        f"'{NOT_FOUND_SENTINEL}'\n"
        f"5. Responda SEMPRE em português brasileiro (pt-BR)."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
