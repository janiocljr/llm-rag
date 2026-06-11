import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Optional

import pdfplumber

try:
    import camelot

    _HAS_CAMELOT = True
except ImportError:
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
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(self, max_tokens: int, overlap_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._max_chars = max_tokens * 4
        self._overlap_chars = overlap_tokens * 4

    def split(self, text: str) -> list[str]:
        chunks: list[str] = []
        self._split_recursive(text, self.SEPARATORS, chunks)
        return [c for c in chunks if c.strip()]

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
        result: list[str],
    ) -> None:
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
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        index_dir: Optional[Path] = None,
        use_semantic_chunking: bool = True,
        remove_headers_footers: bool = True,
    ) -> None:
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

                    if df.shape[1] < 2:
                        logger.debug(
                            f"Skipping single-column false-positive table on page {page_num}"
                        )
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

    @staticmethod
    def _format_pdfplumber_table(table: list[list]) -> str:
        rows = []
        for row in table:
            cells = [(cell or "").strip().replace("\n", " ") for cell in row]
            if any(cells):
                rows.append(cells)
        if not rows:
            return ""

        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]
        keep = [j for j in range(n_cols) if any(r[j] for r in rows)]
        if not keep:
            return ""
        return "\n".join(" | ".join(r[j] for j in keep) for r in rows)

    def _extract_tables_with_pdfplumber(
        self, path: Path, page_num: int, page
    ) -> list[tuple[str, Optional[str]]]:
        results = []

        try:
            tables = page.extract_tables()
            if not tables:
                return results

            for t_idx, table in enumerate(tables):
                if not table:
                    continue

                try:
                    table_str = self._format_pdfplumber_table(table)
                    if not table_str:
                        continue

                    df = pd.DataFrame(table)
                    csv_path = self._save_table_csv(path, page_num, t_idx, df)
                    results.append((table_str, csv_path or None))
                except Exception as e:
                    logger.warning(f"Error processing pdfplumber table {t_idx}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")

        return results

    def _extract_tables_from_page(
        self, path: Path, page_num: int, page
    ) -> list[tuple[str, Optional[str]]]:
        extracted = self._extract_tables_with_camelot(path, page_num)
        if not extracted:
            extracted = self._extract_tables_with_pdfplumber(path, page_num, page)

        results = []
        for t_idx, (table_text, csv_path) in enumerate(extracted):
            if len(table_text.strip()) < 60:
                continue
            header = f"[Tabela {t_idx + 1} — página {page_num} de {path.name}]\n"
            results.append((header + table_text, csv_path))
        return results

    def _format_table_text(self, df: pd.DataFrame) -> str:
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
        if not self._tables_dir:
            return ""

        try:
            filename = f"{path.stem}_p{page_num}_t{table_idx}.csv"
            csv_path = self._tables_dir / filename
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.debug(f"Saved table to: {csv_path}")
            return str(Path("tables") / filename)
        except Exception as e:
            logger.warning(f"Failed to save table CSV: {e}")
            return ""

    def _process_page(
        self, path: Path, page_idx: int, page, filename: str
    ) -> Iterator[DocumentChunk]:
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""

        if self.remove_headers_footers:
            raw_text = self.header_detector.remove_from_text(raw_text)

        import re as _re
        footnote_pattern = _re.compile(r"^(\d{1,2})\s+([A-ZÁÉÍÓÚ][^\n]{20,})", _re.MULTILINE)
        footnote_chunks: list[str] = []
        def _pull_footnotes(text: str) -> str:
            cleaned_lines = []
            for line in text.split("\n"):
                m = footnote_pattern.match(line)
                if m:
                    footnote_chunks.append(f"[Nota {m.group(1)}] {m.group(2)}")
                else:
                    cleaned_lines.append(line)
            return "\n".join(cleaned_lines)

        prose_text = _pull_footnotes(raw_text)

        text_chunks: list[str] = []
        if prose_text.strip():
            cleaned_text = clean_text(prose_text)
            text_chunks = (
                self.semantic_splitter.split(cleaned_text)
                if self.use_semantic_chunking
                else self.splitter.split(cleaned_text)
            )

        text_chunks.extend(footnote_chunks)

        table_entries = self._extract_tables_from_page(path, page_num, page)
        table_chunks: list[tuple[str, Optional[str]]] = []
        for table_text, csv_path in table_entries:
            cleaned_table = clean_text(table_text)
            for piece in self.splitter.split(cleaned_table):
                table_chunks.append((piece, csv_path))

        all_entries: list[tuple[str, bool, Optional[str]]] = [
            (text, False, None) for text in text_chunks
        ] + [(text, True, csv_path) for text, csv_path in table_chunks]

        if not all_entries:
            logger.debug(f"Page {page_num} has no content")
            return

        logger.debug(
            f"Page {page_num}: {len(text_chunks)} text/footnote chunks + {len(table_chunks)} table chunks"
        )

        for chunk_idx, (chunk_text, is_table, csv_path) in enumerate(all_entries):
            if not chunk_text.strip():
                continue

            yield DocumentChunk(
                chunk_id=f"{path.stem}_p{page_num}_c{chunk_idx}",
                text=chunk_text,
                source_file=filename,
                page_number=page_num,
                chunk_index=chunk_idx,
                total_chunks_in_page=len(all_entries),
                char_count=len(chunk_text),
                token_estimate=estimate_tokens(chunk_text),
                is_table=is_table,
                table_csv_path=csv_path,
                semantic_type="table" if is_table else self._infer_semantic_type(chunk_text),
            )

    def load_pdf(self, path: Path) -> Iterator[DocumentChunk]:
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
