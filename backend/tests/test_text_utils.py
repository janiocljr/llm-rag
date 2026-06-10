import pytest

from app.core.text_utils import clean_text, estimate_tokens


class TestCleanText:

    def test_clean_text_basic(self) -> None:
        text = "Hello world"
        assert clean_text(text) == "Hello world"

    def test_clean_text_removes_leading_trailing_whitespace(self) -> None:
        text = "  \n  Hello world  \n  "
        assert clean_text(text) == "Hello world"

    def test_clean_text_removes_multiple_spaces(self) -> None:
        text = "Hello    world"
        assert clean_text(text) == "Hello world"

    def test_clean_text_removes_multiple_newlines(self) -> None:
        text = "Line 1\n\n\n\nLine 2"
        result = clean_text(text)
        assert "\n\n" in result
        assert "\n\n\n" not in result

    def test_clean_text_preserves_double_newlines(self) -> None:
        text = "Paragraph 1\n\nParagraph 2"
        assert clean_text(text) == "Paragraph 1\n\nParagraph 2"

    def test_clean_text_rejoins_hyphenated_words(self) -> None:
        text = "This is a hyphen-\nated word"
        result = clean_text(text)
        assert result == "This is a hyphenated word"

    def test_clean_text_removes_page_numbers(self) -> None:
        text = "Some text\n42\nMore text"
        result = clean_text(text)
        assert "42" not in result

    def test_clean_text_handles_ligatures(self) -> None:
        text = "ﬁnance and ﬂow"
        result = clean_text(text)
        assert isinstance(result, str)
        assert "nance" in result
        assert "flow" in result or "ﬂow" in result

    def test_clean_text_complex_example(self) -> None:
        text = """
        Chapter 1


        This is a para-
        graph that spans mul-
        tiple lines.

        42

        Another    paragraph   with   spaces.
        """
        result = clean_text(text)
        assert "paragraph" in result
        assert "\n\n\n" not in result
        assert "42" not in result
        assert "    " not in result

    def test_clean_text_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_clean_text_only_whitespace(self) -> None:
        assert clean_text("   \n\n   ") == ""

    def test_clean_text_accented_characters(self) -> None:
        text = "Português e São Paulo"
        result = clean_text(text)
        assert "Português" in result
        assert "São Paulo" in result


class TestEstimateTokens:

    def test_estimate_tokens_basic(self) -> None:
        text = "Hello world"
        tokens = estimate_tokens(text)
        assert tokens == 2

    def test_estimate_tokens_empty_string(self) -> None:
        tokens = estimate_tokens("")
        assert tokens == 1

    def test_estimate_tokens_single_char(self) -> None:
        tokens = estimate_tokens("a")
        assert tokens == 1

    def test_estimate_tokens_is_positive(self) -> None:
        for i in range(10):
            tokens = estimate_tokens("a" * i)
            assert tokens >= 1

    def test_estimate_tokens_proportional_to_length(self) -> None:
        text1 = "a" * 100
        text2 = "a" * 200
        tokens1 = estimate_tokens(text1)
        tokens2 = estimate_tokens(text2)
        assert tokens2 > tokens1

    def test_estimate_tokens_long_text(self) -> None:
        text = "This is a sample document. " * 100
        tokens = estimate_tokens(text)
        expected = len(text) // 4
        assert tokens == max(1, expected)

    def test_estimate_tokens_typical_chunk_size(self) -> None:
        chunk = "This is a paragraph about some topic. " * 10
        tokens = estimate_tokens(chunk)
        assert 50 < tokens < 150

    def test_estimate_tokens_with_newlines(self) -> None:
        text = "Line 1\nLine 2\nLine 3\n" * 10
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_estimate_tokens_unicode_characters(self) -> None:
        text = "Portuguese: São Paulo e Brasília" * 5
        tokens = estimate_tokens(text)
        assert tokens > 0
