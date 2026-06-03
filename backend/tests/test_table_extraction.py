import os
from pathlib import Path

import pandas as pd
import pytest


class FakePage:
    def __init__(self, text="Page text", tables=None):
        self._text = text
        self._tables = tables or []

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return self._tables


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def make_fake_table_df():
    return pd.DataFrame([{"A": "1", "B": "2"}, {"A": "3", "B": "4"}])


def test_camelot_preferred_and_csv_saved(monkeypatch, tmp_path):
    """When Camelot finds tables, they are saved as CSV and table chunks contain metadata."""
    from app.core.ingestion import PDFIngester, _HAS_CAMELOT


    monkeypatch.setattr('app.core.ingestion._HAS_CAMELOT', True)


    fake_pdf = FakePDF([FakePage(text="Some text", tables=[])])
    monkeypatch.setattr('pdfplumber.open', lambda p: fake_pdf)


    class FakeTable:
        def __init__(self, df):
            self.df = df

    class FakeTables:
        def __init__(self, tables):
            self._tables = tables

        @property
        def n(self):
            return len(self._tables)

        def __iter__(self):
            return iter(self._tables)

    def fake_read_pdf(path, pages, flavor):
        return FakeTables([FakeTable(make_fake_table_df())])

    monkeypatch.setattr('camelot.read_pdf', fake_read_pdf)


    pdf_dir = tmp_path / "backend" / "data" / "pdfs"
    csv_index_dir = tmp_path / "backend" / "data" / "index" / "tables"
    csv_index_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "test_tables.pdf"
    pdf_path.write_bytes(b"%%PDF-1.4\n%fakepdf")


    index_dir = tmp_path / "backend" / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    ing = PDFIngester(index_dir=index_dir)
    chunks = list(ing.load_pdf(pdf_path))


    table_chunks = [c for c in chunks if getattr(c, 'is_table', False)]
    assert table_chunks, "No table chunk produced"
    for tc in table_chunks:
        assert tc.table_csv_path is not None

        assert not Path(tc.table_csv_path).is_absolute(), f"Expected relative path, got: {tc.table_csv_path}"

        assert (index_dir / tc.table_csv_path).exists(), f"CSV not found: {index_dir / tc.table_csv_path}"


def test_pdfplumber_fallback_when_camelot_fails(monkeypatch, tmp_path):
    """If Camelot finds nothing or raises, pdfplumber.extract_tables is used."""
    from app.core.ingestion import PDFIngester


    monkeypatch.setattr('app.core.ingestion._HAS_CAMELOT', True)


    sample_table = [["a", "b"], ["c", "d"]]
    fake_pdf = FakePDF([FakePage(text="Page text", tables=[sample_table])])
    monkeypatch.setattr('pdfplumber.open', lambda p: fake_pdf)


    def fake_read_pdf_raise(path, pages, flavor):
        raise RuntimeError("camelot failure simulated")

    monkeypatch.setattr('camelot.read_pdf', fake_read_pdf_raise)

    pdf_dir = tmp_path / "backend" / "data" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "test_plumber.pdf"
    pdf_path.write_bytes(b"%%PDF-1.4\n%fakepdf")

    ing = PDFIngester(index_dir=tmp_path / "backend" / "data" / "index")
    chunks = list(ing.load_pdf(pdf_path))


    found = any("pdfplumber" in c.text for c in chunks)
    assert found, "pdfplumber fallback text not found in any chunk"
