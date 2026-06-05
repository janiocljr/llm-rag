"""
app/core/advanced_ingestion.py
==============================
Advanced PDF processing features:
1. Header/Footer Detection & Removal
2. Semantic Paragraph Chunking
3. Vectorized Document Headers
4. LLM-Assisted Indexing (metadata generation)
5. Semantic Pseudo-Documents (chunk grouping)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import pdfplumber
import numpy as np

from app.models.schemas import DocumentChunk
from app.core.text_utils import estimate_tokens, clean_text

logger = logging.getLogger(__name__)


@dataclass
class HeaderFooterMarkers:
    """Detected header/footer patterns in a document."""
    header_pattern: Optional[str] = None
    footer_pattern: Optional[str] = None
    header_height_pct: float = 0.15
    footer_height_pct: float = 0.15


@dataclass
class SemanticHeader:
    """Vectorized summary header for a document."""
    document_name: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    document_type: str = ""
    page_count: int = 0
    chunk_count: int = 0


@dataclass
class SemanticPseudoDocument:
    """Logical grouping of related chunks."""
    pseudo_doc_id: str
    title: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    page_range: Optional[tuple[int, int]] = None

    def to_chunk(self) -> DocumentChunk:
        """Convert pseudo-document to a queryable chunk."""
        combined_text = f"# {self.title}\n\n{self.summary}\n\n" + "\n\n---\n\n".join(
            c.text for c in self.chunks
        )
        return DocumentChunk(
            chunk_id=self.pseudo_doc_id,
            text=combined_text,
            source_file=self.chunks[0].source_file if self.chunks else "",
            page_number=self.chunks[0].page_number if self.chunks else 1,
            chunk_index=-1,
            total_chunks_in_page=1,
            char_count=len(combined_text),
            token_estimate=estimate_tokens(combined_text),
        )


class HeaderFooterDetector:
    """Detects and removes headers/footers from PDF pages."""

    def __init__(self):
        self.markers = HeaderFooterMarkers()
        self.margin_ratio = 0.15

    def detect_from_pages(self, pdf_pages: list) -> HeaderFooterMarkers:
        """Analyze multiple pages to detect header/footer patterns."""
        if len(pdf_pages) < 3:
            return self.markers

        header_lines = []
        footer_lines = []

        for page in pdf_pages[:min(5, len(pdf_pages))]:
            bbox = page.bbox
            page_height = bbox[3] - bbox[1]

            header_bbox = (
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[1] + page_height * self.margin_ratio,
            )
            footer_bbox = (
                bbox[0],
                bbox[3] - page_height * self.margin_ratio,
                bbox[2],
                bbox[3],
            )

            header_crop = page.crop(header_bbox)
            footer_crop = page.crop(footer_bbox)

            header_text = (header_crop.extract_text() or "").strip()
            footer_text = (footer_crop.extract_text() or "").strip()

            if header_text:
                header_lines.append(header_text)
            if footer_text:
                footer_lines.append(footer_text)

        if header_lines:
            common_header = self._find_common_pattern(header_lines)
            if common_header:
                self.markers.header_pattern = common_header

        if footer_lines:
            common_footer = self._find_common_pattern(footer_lines)
            if common_footer:
                self.markers.footer_pattern = common_footer

        return self.markers

    def _find_common_pattern(self, lines: list[str], threshold: float = 0.5) -> Optional[str]:
        """Find common text pattern across lines."""
        if not lines:
            return None

        if len(lines) == 1:
            return lines[0] if len(lines[0]) < 100 else None

        similar_count = {}
        for line in lines:
            for other_line in lines:
                if self._similarity(line, other_line) > threshold:
                    similar_count[line] = similar_count.get(line, 0) + 1

        if similar_count:
            return max(similar_count, key=similar_count.get)
        return None

    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        """Simple string similarity using character overlap."""
        if not s1 or not s2:
            return 0.0
        common = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
        return common / max(len(s1), len(s2))

    def remove_from_text(self, text: str) -> str:
        """Remove detected header/footer patterns from text."""
        if self.markers.header_pattern:
            text = text.replace(self.markers.header_pattern, "")
        if self.markers.footer_pattern:
            text = text.replace(self.markers.footer_pattern, "")
        return text.strip()


class SemanticParagraphChunker:
    """
    Intelligent paragraph-based chunking that respects semantic boundaries.
    Looks for:
    - Blank lines (paragraph breaks)
    - Headings (# or patterns)
    - Lists
    - Natural sentence breaks
    """

    HEADING_PATTERN = re.compile(r"^(#{1,6}\s+|[A-Z][A-Z\s]{3,}:\s*$)", re.MULTILINE)
    LIST_PATTERN = re.compile(r"^\s*[-•*]\s+|\d+\.\s+", re.MULTILINE)

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.max_chars = max_tokens * 4
        self.overlap_chars = overlap_tokens * 4

    def split(self, text: str) -> list[str]:
        """Split text into semantic chunks."""
        paragraphs = self._extract_paragraphs(text)
        chunks = self._merge_paragraphs(paragraphs)
        return chunks

    def _extract_paragraphs(self, text: str) -> list[tuple[str, int]]:
        """Extract paragraphs with their semantic importance."""
        paragraphs = []
        current_para = ""
        importance = 0

        for line in text.split("\n"):
            line_stripped = line.strip()

            if not line_stripped:
                if current_para.strip():
                    paragraphs.append((current_para.strip(), importance))
                    current_para = ""
                    importance = 0
                continue

            is_heading = bool(self.HEADING_PATTERN.match(line_stripped))
            is_list = bool(self.LIST_PATTERN.match(line_stripped))

            if is_heading:
                if current_para.strip():
                    paragraphs.append((current_para.strip(), importance))
                    current_para = ""
                importance = 3
                current_para = line_stripped
            elif is_list:
                if importance < 2:
                    importance = 2
                current_para += "\n" + line if current_para else line
            else:
                current_para += "\n" + line if current_para else line

        if current_para.strip():
            paragraphs.append((current_para.strip(), importance))

        return paragraphs

    def _merge_paragraphs(self, paragraphs: list[tuple[str, int]]) -> list[str]:
        """Merge small paragraphs into chunks respecting token limits."""
        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para_text, importance in paragraphs:
            para_tokens = estimate_tokens(para_text)

            if current_tokens + para_tokens <= self.max_tokens:
                current_chunk += "\n\n" + para_text if current_chunk else para_text
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para_text
                current_tokens = para_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if c.strip()]


class VectorizedHeaderGenerator:
    """Generates semantic headers/summaries for documents."""

    def __init__(self, llm=None):
        self.llm = llm

    def generate_from_first_page(
        self, text: str, filename: str, llm_context=None
    ) -> SemanticHeader:
        """Generate header from first page content."""
        lines = text.split("\n")[:50]
        first_page_text = "\n".join(lines)

        keywords = self._extract_keywords(first_page_text)
        doc_type = self._infer_document_type(first_page_text)

        summary = first_page_text[:500]

        return SemanticHeader(
            document_name=filename,
            summary=summary,
            keywords=keywords,
            document_type=doc_type,
        )

    def _extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """Extract top keywords from text."""
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        from collections import Counter

        word_freq = Counter(words)
        stop_words = {
            "para",
            "como",
            "pelo",
            "pelo",
            "pela",
            "sobre",
            "este",
            "este",
            "that",
            "with",
            "from",
            "which",
        }

        keywords = [
            word for word, _ in word_freq.most_common(top_n * 3) if word not in stop_words
        ]
        return keywords[:top_n]

    def _infer_document_type(self, text: str) -> str:
        """Infer document type from content."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["annual report", "relatório anual"]):
            return "annual_report"
        elif any(w in text_lower for w in ["financial statement", "demonstração financeira"]):
            return "financial_statement"
        elif any(w in text_lower for w in ["contract", "contrato"]):
            return "contract"
        elif any(w in text_lower for w in ["meeting", "reunião", "ata"]):
            return "meeting_minutes"
        else:
            return "general_document"


class SemanticDocumentGrouper:
    """Groups related chunks into logical pseudo-documents."""

    def __init__(self, embedder=None, threshold: float = 0.7):
        self.embedder = embedder
        self.threshold = threshold

    def group_chunks(self, chunks: list[DocumentChunk]) -> list[SemanticPseudoDocument]:
        """Group chunks into semantic pseudo-documents."""
        if not chunks or not self.embedder:
            return self._group_by_page(chunks)

        return self._group_by_similarity(chunks)

    def _group_by_page(self, chunks: list[DocumentChunk]) -> list[SemanticPseudoDocument]:
        """Simple grouping by page number."""
        grouped = {}

        for chunk in chunks:
            key = f"{chunk.source_file}_p{chunk.page_number}"
            if key not in grouped:
                grouped[key] = SemanticPseudoDocument(
                    pseudo_doc_id=key,
                    title=f"{chunk.source_file} - Page {chunk.page_number}",
                    page_range=(chunk.page_number, chunk.page_number),
                )
            grouped[key].chunks.append(chunk)

        return list(grouped.values())

    def _group_by_similarity(self, chunks: list[DocumentChunk]) -> list[SemanticPseudoDocument]:
        """Group by semantic similarity of chunks."""
        if not chunks:
            return []

        embeddings = self.embedder.embed_documents(chunks)
        embeddings = np.array(embeddings)

        groups = []
        assigned = set()

        for i, chunk in enumerate(chunks):
            if i in assigned:
                continue

            group_chunks = [chunk]
            assigned.add(i)

            for j, other_chunk in enumerate(chunks):
                if j <= i or j in assigned:
                    continue

                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )

                if similarity > self.threshold:
                    group_chunks.append(other_chunk)
                    assigned.add(j)

            page_nums = [c.page_number for c in group_chunks]
            pseudo_doc = SemanticPseudoDocument(
                pseudo_doc_id=f"pseudo_{chunk.source_file}_{i}",
                title=f"Group: {chunks[0].source_file}",
                chunks=group_chunks,
                page_range=(min(page_nums), max(page_nums)),
            )
            groups.append(pseudo_doc)

        return groups
