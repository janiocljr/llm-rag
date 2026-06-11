"""Testes do guard contra falso "não encontrei" (classify_answer e streaming)."""
from app.core.llm import (
    NOT_FOUND_SENTINEL,
    classify_answer,
    stream_with_false_negative_guard,
)


class TestClassifyAnswer:

    def test_clean_answer_is_found(self):
        raw = "O IPCA alcançou 5,35% [Fonte 1, Página 6]."
        text, found = classify_answer(raw)
        assert found is True
        assert text == raw

    def test_genuine_not_found(self):
        text, found = classify_answer(NOT_FOUND_SENTINEL)
        assert found is False
        assert text == NOT_FOUND_SENTINEL

    def test_false_negative_prefix_is_stripped(self):
        raw = (
            NOT_FOUND_SENTINEL
            + " Segundo [Fonte 2, Página 6], o IPCA alcançou 5,35% nos doze "
            "meses terminados em junho de 2025."
        )
        text, found = classify_answer(raw)
        assert found is True
        assert not text.startswith(NOT_FOUND_SENTINEL)
        assert "5,35%" in text

    def test_sentinel_mid_text_with_citation_is_found(self):
        raw = "O documento informa 5,35% [Fonte 2, Página 6]. " + NOT_FOUND_SENTINEL
        text, found = classify_answer(raw)
        assert found is True

    def test_short_tail_after_sentinel_stays_not_found(self):
        raw = NOT_FOUND_SENTINEL + " Desculpe."
        text, found = classify_answer(raw)
        assert found is False

    def test_elaborate_negative_without_citation_stays_not_found(self):
        # Caso real: o modelo reformula a negativa com uma explicação longa,
        # mas sem citar fonte — continua sendo "não encontrado" genuíno.
        raw = (
            NOT_FOUND_SENTINEL
            + " Os trechos fornecidos não contêm dados específicos sobre a "
            "população exata de Curitiba em 2025."
        )
        text, found = classify_answer(raw)
        assert found is False
        assert text == raw


class TestStreamGuard:

    def test_normal_stream_passes_through(self):
        tokens = ["Olá", " mundo", "!"]
        assert "".join(stream_with_false_negative_guard(tokens)) == "Olá mundo!"

    def test_strips_sentinel_prefix_when_content_follows(self):
        tokens = [
            NOT_FOUND_SENTINEL[:20],
            NOT_FOUND_SENTINEL[20:],
            " Segundo [Fonte 1, Página 3], o valor foi de 5,35% em junho de 2025.",
        ]
        out = "".join(stream_with_false_negative_guard(tokens))
        assert NOT_FOUND_SENTINEL not in out
        assert "5,35%" in out

    def test_genuine_not_found_is_preserved(self):
        out = "".join(stream_with_false_negative_guard([NOT_FOUND_SENTINEL]))
        assert out == NOT_FOUND_SENTINEL

    def test_elaborate_negative_stream_without_citation_is_kept(self):
        tokens = [
            NOT_FOUND_SENTINEL,
            " Os trechos fornecidos não contêm dados específicos sobre a "
            "população exata de Curitiba em 2025.",
        ]
        out = "".join(stream_with_false_negative_guard(tokens))
        assert out.startswith(NOT_FOUND_SENTINEL)
        assert "Curitiba" in out

    def test_empty_stream_yields_nothing(self):
        assert list(stream_with_false_negative_guard([])) == []
