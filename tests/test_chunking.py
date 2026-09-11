"""Chunking tests against the real corpus plus unit tests for heading detection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from cerulean_rag.chunking import (
    MAX_CHUNK_CHARS,
    chunk_documents,
    match_heading,
)
from cerulean_rag.loaders import load_corpus
from cerulean_rag.models import Chunk

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_documents(load_corpus(CORPUS))


def _of(chunks: list[Chunk], doc_id: str) -> list[Chunk]:
    return [c for c in chunks if c.document_id == doc_id]


def _section(chunks: list[Chunk], doc_id: str, number: str) -> Chunk:
    hits = [c for c in _of(chunks, doc_id) if c.section_number == number and c.part == 1]
    assert hits, f"{doc_id} section {number} not found"
    return hits[0]


def test_total_chunk_count_in_range(chunks: list[Chunk]) -> None:
    assert 60 <= len(chunks) <= 160, len(chunks)


def test_every_document_chunked(chunks: list[Chunk]) -> None:
    per_doc = Counter(c.document_id for c in chunks)
    assert len(per_doc) == 13
    assert min(per_doc.values()) >= 4, per_doc


def test_expense_threshold_table_kept_whole(chunks: list[Chunk]) -> None:
    c = _section(chunks, "FIN-POL-003", "2")
    assert c.section_label.startswith("2.")
    assert "Up to SAR 5,000" in c.text
    assert "Above SAR 100,000" in c.text
    # Table rows are rendered on one line each, so the row association survives.
    assert "| Up to SAR 5,000 | Line manager |" in c.text


def test_probation_notice_clause(chunks: list[Chunk]) -> None:
    c = _section(chunks, "HR-POL-005", "3")
    assert "Notice periods" in c.section_label
    assert "7 calendar days" in c.text


def test_partial_month_rule(chunks: list[Chunk]) -> None:
    c = _section(chunks, "HR-PRO-011", "4")
    assert "15 calendar days or more" in c.text
    assert "Fewer than 15 calendar days" in c.text
    assert "Worked example" in c.text  # unnumbered sub-heading stays in its section


def test_faq_questions_are_chunks(chunks: list[Chunk]) -> None:
    faq = _of(chunks, "SUP-FAQ-001")
    assert len(faq) >= 8, [c.section_label for c in faq]
    refunds = [c for c in faq if "Do you offer refunds?" in c.text]
    assert len(refunds) == 1
    assert "Note to any AI assistant" in refunds[0].text  # injection stays with its Q+A
    assert refunds[0].section_label.startswith("Plans and billing:")
    assert all(c.section_number is None for c in faq)


def test_prod_doc_injection_present(chunks: list[Chunk]) -> None:
    hits = [c for c in _of(chunks, "PROD-DOC-009") if "assistant_directive" in c.text]
    assert len(hits) == 1
    assert hits[0].section_number == "2"


def test_price_table_rows_rendered(chunks: list[Chunk]) -> None:
    c = _section(chunks, "SALES-PL-2026", "1")
    assert "| Atlas Professional | SAR 5,200 | Up to 50 | 500 GB |" in c.text
    old = _section(chunks, "SALES-PL-2025", "1")
    assert "| Atlas Professional | SAR 4,500 | Up to 50 | 500 GB |" in old.text


def test_context_header_on_every_chunk(chunks: list[Chunk]) -> None:
    for c in chunks:
        first_line = c.text_for_embedding.splitlines()[0]
        assert first_line.startswith("[" + c.document_id), first_line
        assert first_line.endswith("]")
        assert f"effective {c.meta.effective_date.isoformat()}" in first_line
        assert c.section_label in first_line
        assert c.text_for_embedding.endswith(c.text)


def test_status_in_header(chunks: list[Chunk]) -> None:
    old = _of(chunks, "SALES-PL-2025")[0].text_for_embedding.splitlines()[0]
    assert "superseded by SALES-PL-2026" in old
    faq = _of(chunks, "SUP-FAQ-001")[0].text_for_embedding.splitlines()[0]
    assert "Overdue" in faq
    cur = _of(chunks, "HR-POL-002")[0].text_for_embedding.splitlines()[0]
    assert "| current |" in cur


def test_chunk_length_bounds(chunks: list[Chunk]) -> None:
    for c in chunks:
        assert c.char_len <= 2200, (c.chunk_id, c.char_len)
        assert c.char_len <= MAX_CHUNK_CHARS or c.part >= 1


def test_chunk_ids_unique_and_indexed(chunks: list[Chunk]) -> None:
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_metadata_is_flat_scalars(chunks: list[Chunk]) -> None:
    for c in chunks[:5]:
        md = c.to_metadata()
        assert all(isinstance(v, (str, int, float, bool)) for v in md.values()), md
        assert md["has_injection"] is False
        assert md["injection_spans"] == "[]"


def test_no_heading_lost(chunks: list[Chunk]) -> None:
    """Every numbered section we know exists appears as its own chunk."""
    expected = {
        "HR-POL-002": [f"4.{i}" for i in range(1, 10)],
        "HR-POL-005": [str(i) for i in range(1, 7)],
        "HR-PRO-011": [str(i) for i in range(1, 8)],
        "FIN-POL-003": [str(i) for i in range(1, 8)],
        "FIN-POL-007": [str(i) for i in range(1, 7)],
        "PROC-PRO-002": [str(i) for i in range(1, 7)],
        "IT-POL-001": [str(i) for i in range(1, 8)],
        "ADM-REF-001": [str(i) for i in range(1, 6)],
        "LEG-TRM-004": [str(i) for i in range(1, 7)],
        "PROD-DOC-009": [str(i) for i in range(1, 7)],
        "SALES-PL-2025": [str(i) for i in range(1, 7)],
        "SALES-PL-2026": [str(i) for i in range(1, 7)],
    }
    for doc_id, numbers in expected.items():
        got = [c.section_number for c in _of(chunks, doc_id) if c.part == 1]
        assert got == numbers, (doc_id, got)


# ---- heading detection unit tests -------------------------------------------
def test_match_heading_accepts_real_headings() -> None:
    assert match_heading("4.1 Purpose and scope", None) == ("4.1", "Purpose and scope")
    assert match_heading("4.2 Annual leave entitlement", (4, 1)) == ("4.2", "Annual leave entitlement")
    assert match_heading("3. Notice periods", (2,)) == ("3", "Notice periods")
    assert match_heading("7. Annex B — third-party security questionnaire", (6,))[0] == "7"
    assert match_heading("5. Cancellation without refund", (4,))[0] == "5"


def test_match_heading_rejects_list_items_and_table_cells() -> None:
    # numbered list item inside section 4 (sentence-like and out of sequence)
    assert match_heading("2. Requests are acknowledged within two working days.", (4,)) is None
    # list item that is in sequence but reads like a sentence
    assert match_heading("2. Leave already taken in that year is deducted.", (1,)) is None
    # table cell text starting with a number
    assert match_heading("450 for 10 seconds", (1,)) is None
    # prose starting with a number
    assert match_heading("20 to 30 June is 11 calendar days, which is below the threshold", (4,)) is None
