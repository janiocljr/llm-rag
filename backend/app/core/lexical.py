"""Índice lexical BM25 (Okapi) em Python puro — sem dependências externas.

Complementa a busca vetorial: embeddings semânticos são fracos para
correspondência exata de números, datas e siglas ("julho de 2025",
"IPCA", "5,35%"), exatamente o tipo de termo que aparece em perguntas
factuais sobre documentos econômicos. O pipeline funde os dois rankings
via Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords mínimas de PT-BR: apenas palavras funcionais muito frequentes.
# Lista curta de propósito — remover demais prejudica perguntas curtas.
_STOPWORDS = frozenset(
    "a o e de da do das dos em um uma uns umas para com que no na nos nas "
    "por se ao aos as os sua seu suas seus foi sao ser esta este isso essa "
    "esse como mais entre sobre qual quais quando onde".split()
)


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acentos, tokens alfanuméricos, sem stopwords."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_len = 0.0

    @property
    def size(self) -> int:
        return len(self._doc_freqs)

    def build(self, texts: list[str]) -> None:
        self._doc_freqs = []
        self._doc_lens = []
        df: Counter = Counter()

        for text in texts:
            tokens = tokenize(text)
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            df.update(freqs.keys())

        n_docs = len(texts)
        self._avg_len = (sum(self._doc_lens) / n_docs) if n_docs else 0.0
        self._idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Retorna [(posição_do_documento, score_bm25)] decrescente, score > 0."""
        if not self._doc_freqs:
            return []

        query_terms = [t for t in tokenize(query) if t in self._idf]
        if not query_terms:
            return []

        scores = [0.0] * len(self._doc_freqs)
        for term in query_terms:
            idf = self._idf[term]
            for pos, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term)
                if not tf:
                    continue
                norm = self.k1 * (
                    1 - self.b + self.b * self._doc_lens[pos] / self._avg_len
                )
                scores[pos] += idf * tf * (self.k1 + 1) / (tf + norm)

        ranked = sorted(
            ((pos, s) for pos, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
