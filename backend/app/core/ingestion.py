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

import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import pdfplumber
try:
    import camelot
    _HAS_CAMELOT = True
except Exception:  #pragma: no cover - optional dependency
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
    Splits text into chunks of at most `max_tokens` tokens,
    trying preferred separators in order.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(self, max_tokens: int, overlap_tokens: int):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self._max_chars = max_tokens * 4
        self._overlap_chars = overlap_tokens * 4

    def split(self, text: str) -> list[str]:
        """Return a list of text chunks with overlap."""
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
        Recursively split `text` using the first separator that produces
        pieces small enough. If no separator works, force-split at max_chars.
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

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, index_dir: Path | None = None, use_semantic_chunking: bool = True, remove_headers_footers: bool = True):
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

    def _extract_tables_from_page(self, path: Path, page_num: int, page) -> str:
        """Extract tables from a page, return markdown text."""
        tables_text = ""

        if _HAS_CAMELOT:
            try:
                tables = camelot.read_pdf(str(path), pages=str(page_num), flavor="lattice")
                if not tables or tables.n == 0:
                    tables = camelot.read_pdf(str(path), pages=str(page_num), flavor="stream")

                if tables and tables.n > 0:
                    for t_idx, table in enumerate(tables):
                        try:
                            md = self._format_table(table)
                            tables_text += f"\n[Tabela {t_idx + 1} da página {page_num} — Camelot]:\n{md}\n"
                        except Exception:
                            pass
            except Exception:
                pass

        if not tables_text:
            for t_idx, table in enumerate(page.extract_tables()):
                if not table:
                    continue
                rows = [" | ".join(cell.strip() if cell else "" for cell in row) for row in table]
                table_str = "\n".join(rows)
                tables_text += f"\n[Tabela {t_idx + 1} da página {page_num} — pdfplumber]:\n{table_str}\n"

        return tables_text

    def _format_table(self, table) -> str:
        """Format table to markdown or CSV."""
        df = table.df
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_csv(index=False)

    def _process_page(self, path: Path, page_idx: int, page, filename: str) -> Iterator[DocumentChunk]:
        """Process a single PDF page."""
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""

        if self.remove_headers_footers:
            raw_text = self.header_detector.remove_from_text(raw_text)

        tables_text = self._extract_tables_from_page(path, page_num, page)
        full_text = raw_text + tables_text

        if not full_text.strip():
            return

        cleaned = clean_text(full_text)
        page_chunks = (
            self.semantic_splitter.split(cleaned)
            if self.use_semantic_chunking
            else self.splitter.split(cleaned)
        )

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
        filename = path.name
        logger.info(f"Ingesting: {filename}")

        try:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"  {total_pages} pages found")

                if self.remove_headers_footers:
                    self.header_detector.detect_from_pages(pdf.pages)

                for page_idx, page in enumerate(pdf.pages):
                    yield from self._process_page(path, page_idx, page, filename)

        except Exception as exc:
            logger.error(f"Failed to parse {filename}: {exc}", exc_info=True)
            raise

    def _infer_semantic_type(self, text: str) -> str:
        """Infer semantic type of a chunk."""
        text_lower = text.lower().strip()

        if text_lower.startswith(("#", "##", "###")) or re.match(r"^[A-Z][A-Z\s]{3,}:", text):
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
        """Load all PDFs in `directory` and return a flat list of chunks."""
        pdf_files = sorted(directory.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {directory}")

        all_chunks: list[DocumentChunk] = []
        for pdf_path in pdf_files:
            chunks = list(self.load_pdf(pdf_path))
            logger.info(f"  → {len(chunks)} chunks from {pdf_path.name}")
            all_chunks.extend(chunks)

        logger.info(f"Total chunks ingested: {len(all_chunks)}")
        return all_chunks
