"""Unit tests for utility functions."""
from __future__ import annotations

import pytest

from app.utils.text import clean_text, truncate, word_count

pytestmark = pytest.mark.unit


class TestCleanText:
    def test_collapses_multiple_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_collapses_multiple_newlines(self):
        result = clean_text("line1\n\n\n\nline2")
        assert result == "line1\n\nline2"

    def test_strips_leading_trailing(self):
        assert clean_text("  hello  ") == "hello"

    def test_normalizes_unicode(self):
        # NFKC: ﬁ (ligature) → fi
        assert clean_text("\ufb01le") == "file"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestWordCount:
    def test_basic(self):
        assert word_count("hello world foo") == 3

    def test_empty(self):
        assert word_count("") == 0

    def test_single_word(self):
        assert word_count("hello") == 1


class TestTruncate:
    def test_no_truncation_needed(self):
        assert truncate("hello world", 10) == "hello world"

    def test_truncates_at_limit(self):
        result = truncate("one two three four five", 3)
        assert result == "one two three…"

    def test_exact_limit(self):
        result = truncate("one two three", 3)
        assert result == "one two three"
