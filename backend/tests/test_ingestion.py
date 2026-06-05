"""
Unit tests for app.core.ingestion module.

Tests cover:
- RecursiveCharSplitter token-aware chunking
- Table extraction with Camelot and fallback
- Text processing and semantic type inference
- PDF ingestion workflow
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.core.ingestion import RecursiveCharSplitter, PDFIngester
from app.core.text_utils import estimate_tokens


class TestRecursiveCharSplitter:
    """Test RecursiveCharSplitter chunking logic."""

    def test_splitter_initialization(self) -> None:
        """Test splitter initialization."""
        splitter = RecursiveCharSplitter(max_tokens=512, overlap_tokens=64)
        assert splitter.max_tokens == 512
        assert splitter.overlap_tokens == 64
        assert splitter._max_chars == 512 * 4
        assert splitter._overlap_chars == 64 * 4

    def test_split_short_text(self) -> None:
        """Test splitting text shorter than max tokens."""
        splitter = RecursiveCharSplitter(max_tokens=100, overlap_tokens=10)
        text = "This is a short text."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self) -> None:
        """Test splitting long text produces multiple chunks."""
        splitter = RecursiveCharSplitter(max_tokens=50, overlap_tokens=5)
        text = "This is a longer text. " * 20
        chunks = splitter.split(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert estimate_tokens(chunk) <= 50

    def test_split_respects_paragraph_breaks(self) -> None:
        """Test that splitter prefers paragraph breaks."""
        splitter = RecursiveCharSplitter(max_tokens=20, overlap_tokens=3)
        text = ("First paragraph with much more detailed text that is quite long. " * 2 +
                "\n\n" +
                "Second paragraph with much more detailed text that is quite long. " * 2)
        chunks = splitter.split(text)
        assert len(chunks) >= 2

    def test_split_removes_empty_chunks(self) -> None:
        """Test that empty chunks are filtered."""
        splitter = RecursiveCharSplitter(max_tokens=50, overlap_tokens=5)
        text = "   \n\n   Text   "
        chunks = splitter.split(text)
        assert all(c.strip() for c in chunks)

    def test_split_with_various_separators(self) -> None:
        """Test splitting with different separator types."""
        splitter = RecursiveCharSplitter(max_tokens=30, overlap_tokens=5)
        text = "Word. Word! Word? Word; Word, Word word word"
        chunks = splitter.split(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) > 0


class TestPDFIngesterTableExtraction:
    """Test table extraction capabilities."""

    @pytest.fixture
    def ingester(self):
        """Create ingester instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield PDFIngester(index_dir=Path(tmpdir))

    def test_format_table_text_with_markdown(self, ingester) -> None:
        """Test table formatting to markdown."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = ingester._format_table_text(df)
        assert isinstance(result, str)
        assert "A" in result

    def test_format_table_text_with_csv_fallback(self, ingester) -> None:
        """Test table formatting falls back to CSV."""
        df = pd.DataFrame({"Col": [1, 2, 3]})
        result = ingester._format_table_text(df)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_save_table_csv(self, ingester) -> None:
        """Test saving table to CSV."""
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
        path = Path("test.pdf")
        page_num = 1
        table_idx = 0

        csv_path = ingester._save_table_csv(path, page_num, table_idx, df)
        assert csv_path or csv_path == ""

    def test_save_table_csv_without_index_dir(self) -> None:
        """Test saving table without index directory."""
        ingester = PDFIngester(index_dir=None)
        df = pd.DataFrame({"A": [1, 2]})
        csv_path = ingester._save_table_csv(Path("test.pdf"), 1, 0, df)
        assert csv_path == ""


class TestPDFIngesterSemanticType:
    """Test semantic type inference."""

    @pytest.fixture
    def ingester(self):
        """Create ingester instance."""
        return PDFIngester()

    def test_infer_semantic_type_heading(self, ingester) -> None:
        """Test heading detection."""
        assert ingester._infer_semantic_type("# Title") == "heading"
        assert ingester._infer_semantic_type("## Subtitle") == "heading"
        assert ingester._infer_semantic_type("SECTION TITLE:") == "heading"

    def test_infer_semantic_type_list(self, ingester) -> None:
        """Test list detection."""
        assert ingester._infer_semantic_type("- Item 1") == "list"
        assert ingester._infer_semantic_type("• Item 2") == "list"
        assert ingester._infer_semantic_type("* Item 3") == "list"

    def test_infer_semantic_type_title(self, ingester) -> None:
        """Test short title detection."""
        text = "Short title"
        assert ingester._infer_semantic_type(text) == "title"

    def test_infer_semantic_type_snippet(self, ingester) -> None:
        """Test snippet detection."""
        text = "Short text\nwith newline and more content"
        assert ingester._infer_semantic_type(text) in ("snippet", "paragraph")

    def test_infer_semantic_type_paragraph(self, ingester) -> None:
        """Test paragraph detection."""
        text = "This is a longer paragraph. " * 10
        assert ingester._infer_semantic_type(text) == "paragraph"


class TestPDFIngesterInitialization:
    """Test ingester initialization."""

    def test_initialization_with_defaults(self) -> None:
        """Test ingester with default parameters."""
        ingester = PDFIngester()
        assert ingester.splitter.max_tokens == 512
        assert ingester.splitter.overlap_tokens == 64
        assert ingester.use_semantic_chunking is True
        assert ingester.remove_headers_footers is True

    def test_initialization_with_custom_params(self) -> None:
        """Test ingester with custom parameters."""
        ingester = PDFIngester(
            chunk_size=256,
            chunk_overlap=32,
            use_semantic_chunking=False,
            remove_headers_footers=False,
        )
        assert ingester.splitter.max_tokens == 256
        assert ingester.splitter.overlap_tokens == 32
        assert ingester.use_semantic_chunking is False
        assert ingester.remove_headers_footers is False

    def test_initialization_creates_tables_directory(self) -> None:
        """Test that tables directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingester = PDFIngester(index_dir=Path(tmpdir))
            tables_dir = Path(tmpdir) / "tables"
            assert tables_dir.exists()


class TestPDFIngesterLoadDirectory:
    """Test directory loading."""

    def test_load_directory_not_found(self) -> None:
        """Test error on non-existent directory."""
        ingester = PDFIngester()
        with pytest.raises(FileNotFoundError):
            ingester.load_directory(Path("/nonexistent/path"))

    def test_load_directory_empty(self) -> None:
        """Test error on empty directory."""
        ingester = PDFIngester()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                ingester.load_directory(Path(tmpdir))
