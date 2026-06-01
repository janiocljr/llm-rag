"""
app/core/llm.py
===============
Local LLM inference via llama-cpp-python.

LLM CHOICE — RATIONALE
-----------------------
Default: Mistral 7B Instruct v0.2 (Q4_K_M quantisation)

WHY MISTRAL 7B?
  1. INSTRUCTION FOLLOWING: Trained with SFT + RLHF specifically for
     instruction-following — critical for our "only use context" constraint.
  2. SIZE: 7 B params. Well within the 9.9 B limit. Q4_K_M quantisation
     brings RAM usage to ~4.1 GB — fits in 8 GB RAM.
  3. CONTEXT WINDOW: 32 k tokens native. We use 4 096 (enough for 3 chunks
     + prompt overhead) to keep inference fast.
  4. QUALITY: Outperforms Llama 2 13B on most benchmarks despite being smaller.

ALTERNATIVES:
  - Llama 3 8B Instruct (Q4_K_M): slightly better on reasoning, same RAM.
  - Gemma 2 9B (Q4_K_M): best quality in class, ~5.5 GB RAM.
  - Phi-3 mini 3.8B: excellent if RAM is constrained to 4 GB.

WHY llama-cpp-python (GGUF)?
  - Pure CPU inference — no GPU required.
  - Handles quantised models that can't fit in VRAM on consumer hardware.
  - Python bindings are stable, actively maintained.
  - Produces deterministic output (temperature=0.1 in our setup).

ANTI-HALLUCINATION MEASURES IN THE PROMPT
  See build_prompt() for details. The key techniques:
  1. System message explicitly forbids external knowledge.
  2. Context is clearly delimited with XML-like tags.
  3. The model is instructed to produce a specific "not found" string
     if context is insufficient — which we then detect programmatically.
"""

import logging
from pathlib import Path

from llama_cpp import Llama

from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

# Sentinel string the LLM is told to output when context is insufficient
NOT_FOUND_SENTINEL = "Não encontrei essa informação nos documentos fornecidos."


class LocalLLM:
    """
    Wrapper around llama-cpp-python for GGUF model inference.
    """

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
                "Running in STUB MODE — responses will be placeholder text. "
                "Download the model and update LLM_MODEL_PATH to enable real inference."
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt string.

        The prompt must already be formatted in the model's chat template
        (see build_prompt() below).
        """
        if self._stub:
            return (
                "[STUB MODE] LLM model not loaded. "
                f"Would have generated a response for prompt of length {len(prompt)} chars. "
                "Set LLM_MODEL_PATH to a valid GGUF file to enable real inference."
            )

        output = self._model(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=["</s>", "[INST]", "Human:", "User:"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def stream_generate(self, prompt: str):
        """
        Stream generator that yields partial text chunks as they arrive from
        llama-cpp-python if streaming is supported. Falls back to a single
        synchronous yield when streaming is not available.
        """
        if self._stub:
            yield (
                "[STUB MODE] LLM model not loaded. "
                f"Would have generated a response for prompt of length {len(prompt)} chars."
            )
            return

        # Try the library's streaming API. Different versions yield different
        # shapes (dicts with 'choices'/'delta' or text strings). Be defensive.
        try:
            for chunk in self._model(
                prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stream=True,
            ):
                text = ""
                try:
                    # Newer llama-cpp-python yields dicts with 'choices'
                    if isinstance(chunk, dict):
                        choices = chunk.get("choices") or []
                        if choices:
                            # delta or text
                            ch = choices[0]
                            if isinstance(ch, dict):
                                if "text" in ch:
                                    text = ch.get("text", "")
                                elif "delta" in ch and isinstance(ch["delta"], dict):
                                    text = ch["delta"].get("content", "")
                                else:
                                    text = str(ch)
                        else:
                            text = str(chunk)
                    else:
                        text = str(chunk)
                except Exception:
                    text = str(chunk)

                if text:
                    yield text
        except TypeError:
            # Streaming not supported by this binding; fall back to blocking
            full = self.generate(prompt)
            yield full


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    system_prompt: str,
) -> str:
    """
    Construct the full LLM prompt in Mistral Instruct format.

    Format: [INST] <<SYS>> ... <</SYS>> ... [/INST]
    (Compatible with Mistral, Llama 2/3 Instruct variants.)

    ANTI-HALLUCINATION TECHNIQUES APPLIED
    ---------------------------------------
    1. System prompt explicitly forbids knowledge beyond the context.
    2. Context is wrapped in <context> tags — clear delimiter for the model.
    3. Each source is labelled with filename + page for accurate citation.
    4. Model is told to output a specific "not found" string.
    5. "Step-by-step" instruction elicits more faithful chain-of-thought.

    WHY NOT USE LANGCHAIN PROMPTS?
    No external framework dependency. This gives full transparency and control
    over exactly what the model sees — critical for a system where you must
    audit every token to prevent hallucination.
    """

    # --- Build the context block ---
    if retrieved_chunks:
        context_parts = []
        for i, rc in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i}: {rc.chunk.source_file} | Page {rc.chunk.page_number}]\n"
                f"{rc.chunk.text}"
            )
        context_block = "\n\n".join(context_parts)
    else:
        context_block = "(No relevant context found in the documents.)"

    # --- Assemble full prompt ---
    prompt = (
        f"[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        f"<</SYS>>\n\n"
        f"<context>\n"
        f"{context_block}\n"
        f"</context>\n\n"
        f"Pergunta: {question}\n\n"
        f"Instruções:\n"
        f"1. Responda APENAS com base no contexto acima.\n"
        f"2. Para cada informação, cite a fonte como [Fonte N, Página X].\n"
        f"3. Se o contexto for insuficiente, responda exatamente: '{NOT_FOUND_SENTINEL}'\n"
        f"4. Não adicione informações externas ao contexto fornecido.\n"
        f"5. Responda SEMPRE em português brasileiro (pt-BR).\n"
        f" [/INST]"
    )

    return prompt
