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


_SENTINEL_TRIM = " \n\t.—–-,;:"


def classify_answer(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    if NOT_FOUND_SENTINEL not in text:
        return text, True

    if text.startswith(NOT_FOUND_SENTINEL):
        rest = text[len(NOT_FOUND_SENTINEL):].lstrip(_SENTINEL_TRIM)
        if len(rest) >= 40 and "[Fonte" in rest:
            return rest, True
        return text, False

    return text, "[Fonte" in text


def stream_with_false_negative_guard(token_iter):
    it = iter(token_iter)
    buf = ""

    for token in it:
        buf += token
        s = buf.lstrip()
        if s.startswith(NOT_FOUND_SENTINEL):
            rest = s[len(NOT_FOUND_SENTINEL):]
            for token2 in it:
                rest += token2
            stripped = rest.lstrip(_SENTINEL_TRIM)
            if len(stripped) >= 40 and "[Fonte" in stripped:
                yield stripped
            elif stripped:
                yield NOT_FOUND_SENTINEL + rest
            else:
                yield NOT_FOUND_SENTINEL
            return
        if not NOT_FOUND_SENTINEL.startswith(s):
            yield buf
            yield from it
            return

    if buf:
        yield buf


def context_char_budget(context_length: int, max_new_tokens: int) -> int:
    reserved_tokens = max_new_tokens + 800
    return max(1_000, (context_length - reserved_tokens) * 3)


def build_messages(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    system_prompt: str,
    max_context_chars: int | None = None,
) -> list[dict]:
    if retrieved_chunks:
        context_parts = []
        used_chars = 0
        for i, rc in enumerate(retrieved_chunks, 1):
            text = rc.chunk.text
            if max_context_chars is not None:
                remaining = max_context_chars - used_chars
                if remaining < 200:
                    logger.warning(
                        "Context budget exhausted — dropping chunks %d..%d from prompt",
                        i, len(retrieved_chunks),
                    )
                    break
                if len(text) > remaining:
                    text = text[:remaining] + "…"
            used_chars += len(text)
            context_parts.append(
                f"[Fonte {i}: {rc.chunk.source_file} | Página {rc.chunk.page_number} | "
                f"Relevância: {rc.score:.1%}]\n{text}"
            )
        context_block = "\n\n---\n\n".join(context_parts)
        context_intro = f"Você tem {len(context_parts)} trechos relevantes dos documentos para responder:"
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
        f"4. Se algum trecho contiver dados do TEMA da pergunta — mesmo que de período, "
        f"recorte ou escopo um pouco diferente do perguntado (ex.: mês ou ano vizinho) — "
        f"você DEVE apresentar esses dados, explicitando o período/escopo exato a que se "
        f"referem. É PROIBIDO responder que a informação não foi encontrada nesse caso.\n"
        f"5. Apenas se NENHUM trecho tiver relação alguma com o tema da pergunta, "
        f"responda exatamente: '{NOT_FOUND_SENTINEL}'\n"
        f"6. Responda SEMPRE em português brasileiro (pt-BR)."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
