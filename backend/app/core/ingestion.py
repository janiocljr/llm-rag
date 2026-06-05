"""
app/core/ingestion.py
=====================
PDF loading, text extraction, cleaning and chunking.

CHUNKING STRATEGY — RATIONALE
------------------------------
We use **token-aware recursive character splitting** rather than fixed-char
splitting or sentence splitting for the following reasons:

1. TOKEN AWARENESS
   Embedding models (and LLMs) operate on tokens, not characters.
   A chunk of 512 characters may be 100 or 200 tokens depending on content.
   We estimate tokens as chars/4 (GPT-style BPE heuristic) which is good
   enough for English/Portuguese text without adding a tokeniser dependency.

2. RECURSIVE SPLITTING
   We try to split at paragraph breaks first (\n\n), then sentence breaks (\n),
   then spaces. This preserves semantic units as much as possible and avoids
   cutting words mid-sentence.

3. OVERLAP
   A 64-token overlap means that a sentence starting at the end of chunk N
   also appears at the beginning of chunk N+1. This prevents answers from
   being lost at chunk boundaries.

4. PAGE AWARENESS
   We track page boundaries. Chunks never cross page boundaries — this
   ensures our page citations are always accurate.

ALTERNATIVE CONSIDERED: Sentence-level chunking (NLTK/spaCy).
WHY REJECTED: Sentence detection fails on PDF text with ligatures, hyphenated
words, and tables. The recursive approach is more robust to noisy PDF output.
"""

"""
PDF text extraction, cleaning, and chunking with table detection.

This module handles:
- PDF page-by-page text extraction using pdfplumber
- Table detection and extraction using Camelot (with pdfplumber fallback)
- Text normalization and cleaning
- Token-aware recursive chunking
- Optional semantic-aware chunking
- Header/footer detection and removal
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Optional

import pdfplumber

try:
    import camelot

    _HAS_CAMELOT = True
except ImportError:  # pragma: no cover
    camelot = None
    _HAS_CAMELOT = False

import pandas as pd

from app.models.schemas import DocumentChunk
from app.core.text_utils import estimate_tokens, clean_text
from app.core.advanced_ingestion import (
    HeaderFooterDetector,
    SemanticParagraphChunker,
    VectorizedHeaderGenerator,
)

logger = logging.getLogger(__name__)


class RecursiveCharSplitter:
    """
    Token-aware recursive character splitter for semantic text chunking.

    Splits text into chunks respecting token limits, using preferred separators
    (paragraph breaks, sentences, words) to maintain semantic coherence.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(self, max_tokens: int, overlap_tokens: int) -> None:
        """
        Initialize splitter.

        Args:
            max_tokens: Target maximum tokens per chunk
            overlap_tokens: Token overlap between consecutive chunks
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self._max_chars = max_tokens * 4
        self._overlap_chars = overlap_tokens * 4

    def split(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks with overlap
        """
        chunks: list[str] = []
        self._split_recursive(text, self.SEPARATORS, chunks)
        return [c for c in chunks if c.strip()]

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
        result: list[str],
    ) -> None:
        """
        Recursively split text using preferred separators.

        Tries separators in order. If no separator works, force-splits at
        character boundaries while respecting overlap.

        Args:
            text: Text to split
            separators: Separators to try (in order of preference)
            result: Accumulator for resulting chunks
        """
        if estimate_tokens(text) <= self.max_tokens:
            if result and estimate_tokens(result[-1] + " " + text) <= self.max_tokens:
                result[-1] = result[-1] + " " + text
            else:
                result.append(text)
            return

        if not separators:
            for i in range(0, len(text), self._max_chars - self._overlap_chars):
                result.append(text[i : i + self._max_chars])
            return

        sep = separators[0]
        remaining_seps = separators[1:]
        parts = text.split(sep) if sep else list(text)

        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if estimate_tokens(candidate) <= self.max_tokens:
                current = candidate
            else:
                if current:
                    self._split_recursive(current, remaining_seps, result)
                    overlap_text = current[-self._overlap_chars :]
                    current = (overlap_text + sep + part).strip()
                else:
                    self._split_recursive(part, remaining_seps, result)
                    current = ""

        if current:
            self._split_recursive(current, remaining_seps, result)






class PDFIngester:
    """
    Loads PDF files, extracts text page by page, cleans and chunks it.

    WHY pdfplumber over PyPDF2/pypdf?
    ----------------------------------
    pdfplumber extracts text with proper word-spacing reconstruction and
    handles complex layouts (multi-column, tables) better than pypdf.
    It also gives us bounding-box metadata for future layout-aware chunking.

    ADVANCED FEATURES
    -----------------
    - Header/footer detection and removal
    - Semantic paragraph-based chunking
    - Vectorized document headers
    - Semantic type classification
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        index_dir: Optional[Path] = None,
        use_semantic_chunking: bool = True,
        remove_headers_footers: bool = True,
    ) -> None:
        """
        Initialize PDF ingester.

        Args:
            chunk_size: Target tokens per chunk
            chunk_overlap: Token overlap between chunks
            index_dir: Directory to save extracted tables (CSVs)
            use_semantic_chunking: Use semantic-aware chunking
            remove_headers_footers: Remove headers/footers from pages
        """
        self.splitter = RecursiveCharSplitter(
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
        )
        self.semantic_splitter = SemanticParagraphChunker(
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
        )
        self.header_detector = HeaderFooterDetector()
        self.header_generator = VectorizedHeaderGenerator()
        self._index_dir = Path(index_dir) if index_dir else None
        self.use_semantic_chunking = use_semantic_chunking
        self.remove_headers_footers = remove_headers_footers

        if self._index_dir:
            self._tables_dir = self._index_dir / "tables"
            self._tables_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Tables will be saved to: {self._tables_dir}")
        else:
            self._tables_dir = None

    def _extract_tables_with_camelot(
        self, path: Path, page_num: int
    ) -> list[tuple[str, str]]:
        """
        Extract tables from page using Camelot.

        Tries 'lattice' flavor first (for hand-drawn tables), then 'stream' flavor
        (for computer-generated tables). Returns list of (table_text, csv_path) tuples.

        Args:
            path: PDF file path
            page_num: Page number (1-indexed)

        Returns:
            List of (formatted_table, csv_path) tuples
        """
        if not _HAS_CAMELOT:
            logger.debug("Camelot not available, skipping table extraction")
            return []

        results = []

        try:
            flavors = ["lattice", "stream"]
            tables = None

            for flavor in flavors:
                try:
                    logger.debug(f"Trying Camelot {flavor} flavor for page {page_num}")
                    tables = camelot.read_pdf(
                        str(path),
                        pages=str(page_num),
                        flavor=flavor,
                        suppress_stdout=True,
                    )

                    if tables and tables.n > 0:
                        logger.debug(f"Found {tables.n} tables with {flavor} flavor")
                        break
                except Exception as e:
                    logger.debug(
                        f"Camelot {flavor} failed for page {page_num}: {e}"
                    )
                    continue

            if not tables or tables.n == 0:
                logger.debug(f"No tables found on page {page_num}")
                return []

            for t_idx, table in enumerate(tables):
                try:
                    df = table.df
                    if df.empty:
                        continue

                    formatted_text = self._format_table_text(df)
                    csv_path = self._save_table_csv(path, page_num, t_idx, df)

                    results.append((formatted_text, csv_path))
                    logger.debug(f"Extracted table {t_idx + 1} from page {page_num}")

                except Exception as e:
                    logger.warning(
                        f"Error processing Camelot table {t_idx} on page {page_num}: {e}"
                    )
                    continue

        except Exception as e:
            logger.warning(f"Camelot extraction failed for page {page_num}: {e}")

        return results

    def _extract_tables_with_pdfplumber(
        self, page
    ) -> list[tuple[str, Optional[str]]]:
        """
        Fallback table extraction using pdfplumber.

        Args:
            page: pdfplumber page object

        Returns:
            List of (formatted_table, None) tuples
        """
        results = []

        try:
            tables = page.extract_tables()
            if not tables:
                return results

            for t_idx, table in enumerate(tables):
                if not table:
                    continue

                try:
                    rows = [
                        " | ".join(
                            cell.strip() if cell else "" for cell in row
                        )
                        for row in table
                    ]
                    table_str = "\n".join(rows)
                    results.append((table_str, None))
                except Exception as e:
                    logger.warning(f"Error processing pdfplumber table {t_idx}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")

        return results

    def _extract_tables_from_page(
        self, path: Path, page_num: int, page
    ) -> str:
        """
        Extract tables from a page using Camelot with pdfplumber fallback.

        Returns markdown/text representation of all tables found.

        Args:
            path: PDF file path
            page_num: Page number (1-indexed)
            page: pdfplumber page object

        Returns:
            Formatted string containing all tables from page
        """
        tables_text = ""

        camelot_tables = self._extract_tables_with_camelot(path, page_num)
        for t_idx, (table_text, csv_path) in enumerate(camelot_tables):
            csv_info = f" → {csv_path}" if csv_path else ""
            tables_text += (
                f"\n[Tabela {t_idx + 1} da página {page_num} — Camelot]{csv_info}:\n"
                f"{table_text}\n"
            )

        if not camelot_tables:
            pdfplumber_tables = self._extract_tables_with_pdfplumber(page)
            for t_idx, (table_text, _) in enumerate(pdfplumber_tables):
                tables_text += (
                    f"\n[Tabela {t_idx + 1} da página {page_num} — "
                    f"pdfplumber]:\n{table_text}\n"
                )

        return tables_text

    def _format_table_text(self, df: pd.DataFrame) -> str:
        """
        Format DataFrame to markdown or CSV text.

        Args:
            df: pandas DataFrame

        Returns:
            Formatted table as markdown or CSV string
        """
        try:
            return df.to_markdown(index=False)
        except Exception:
            try:
                return df.to_csv(index=False)
            except Exception:
                return str(df.values)

    def _save_table_csv(
        self, path: Path, page_num: int, table_idx: int, df: pd.DataFrame
    ) -> str:
        """
        Save extracted table as CSV file.

        Args:
            path: PDF file path
            page_num: Page number (1-indexed)
            table_idx: Table index within page (0-indexed)
            df: pandas DataFrame to save

        Returns:
            Path to saved CSV file (relative to tables directory)
        """
        if not self._tables_dir:
            return ""

        try:
            filename = f"{path.stem}_p{page_num}_t{table_idx}.csv"
            csv_path = self._tables_dir / filename
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.debug(f"Saved table to: {csv_path}")
            return str(csv_path)
        except Exception as e:
            logger.warning(f"Failed to save table CSV: {e}")
            return ""

    def _process_page(
        self, path: Path, page_idx: int, page, filename: str
    ) -> Iterator[DocumentChunk]:
        """
        Process a single PDF page.

        Extracts text and tables, cleans, and chunks into DocumentChunk objects.

        Args:
            path: PDF file path
            page_idx: Page index (0-based)
            page: pdfplumber page object
            filename: Original filename for metadata

        Yields:
            DocumentChunk objects for this page
        """
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""

        if self.remove_headers_footers:
            raw_text = self.header_detector.remove_from_text(raw_text)

        tables_text = self._extract_tables_from_page(path, page_num, page)
        full_text = raw_text + tables_text

        if not full_text.strip():
            logger.debug(f"Page {page_num} has no content")
            return

        cleaned = clean_text(full_text)
        page_chunks = (
            self.semantic_splitter.split(cleaned)
            if self.use_semantic_chunking
            else self.splitter.split(cleaned)
        )

        logger.debug(f"Page {page_num}: {len(page_chunks)} chunks created")

        for chunk_idx, chunk_text in enumerate(page_chunks):
            if not chunk_text.strip():
                continue

            yield DocumentChunk(
                chunk_id=f"{path.stem}_p{page_num}_c{chunk_idx}",
                text=chunk_text,
                source_file=filename,
                page_number=page_num,
                chunk_index=chunk_idx,
                total_chunks_in_page=len(page_chunks),
                char_count=len(chunk_text),
                token_estimate=estimate_tokens(chunk_text),
                semantic_type=self._infer_semantic_type(chunk_text),
            )

    def load_pdf(self, path: Path) -> Iterator[DocumentChunk]:
        """
        Load and ingest a single PDF file.

        Args:
            path: Path to PDF file

        Yields:
            DocumentChunk objects from the PDF

        Raises:
            FileNotFoundError: If PDF file not found
            Exception: If PDF parsing fails
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        filename = path.name
        logger.info(f"Ingesting: {filename}")

        try:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"  {total_pages} pages found")

                if self.remove_headers_footers:
                    logger.debug("Detecting headers/footers...")
                    self.header_detector.detect_from_pages(pdf.pages)

                for page_idx, page in enumerate(pdf.pages):
                    logger.debug(f"Processing page {page_idx + 1}/{total_pages}")
                    yield from self._process_page(path, page_idx, page, filename)

        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.exception(f"Failed to parse {filename}")
            raise

    def _infer_semantic_type(self, text: str) -> str:
        """
        Infer semantic type of a chunk.

        Args:
            text: Chunk text

        Returns:
            Semantic type: 'heading', 'list', 'title', 'snippet', or 'paragraph'
        """
        text_lower = text.lower().strip()

        if text_lower.startswith(("#", "##", "###")) or re.match(
            r"^[A-Z][A-Z\s]{3,}:", text
        ):
            return "heading"
        elif re.match(r"^\s*[-•*]\s+", text):
            return "list"
        elif "\n" not in text and len(text.split()) < 20:
            return "title"
        elif len(text) < 100:
            return "snippet"
        else:
            return "paragraph"

    def load_directory(self, directory: Path) -> list[DocumentChunk]:
        """
        Load all PDFs from a directory.

        Args:
            directory: Directory containing PDF files

        Returns:
            List of all DocumentChunk objects from all PDFs

        Raises:
            FileNotFoundError: If directory has no PDF files
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pdf_files = sorted(directory.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {directory}")

        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        all_chunks: list[DocumentChunk] = []
        for pdf_path in pdf_files:
            try:
                chunks = list(self.load_pdf(pdf_path))
                logger.info(f"  → {len(chunks)} chunks from {pdf_path.name}")
                all_chunks.extend(chunks)
            except Exception:
                logger.exception(f"Skipping {pdf_path.name}")
                continue

        logger.info(f"Total chunks ingested: {len(all_chunks)}")
        return all_chunks
