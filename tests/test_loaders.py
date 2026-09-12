"""Loader tests against the real corpus in ./corpus."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pytest

from cerulean_rag.loaders import (
    CorpusLoadError,
    load_corpus,
    parse_date,
    parse_metadata_block,
    strip_running_headers,
)
from cerulean_rag.models import LoadedDocument

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="module")
def docs() -> list[LoadedDocument]:
    return load_corpus(CORPUS)


@pytest.fixture(scope="module")
def by_id(docs: list[LoadedDocument]) -> dict[str, LoadedDocument]:
    return {d.meta.document_id: d for d in docs}


def test_thirteen_documents_matching_manifest(docs: list[LoadedDocument]) -> None:
    manifest = json.loads((CORPUS / "corpus_manifest.json").read_text(encoding="utf-8"))
    expected = sorted(e["document_id"] for e in manifest["documents"])
    assert len(docs) == 13
    assert [d.meta.document_id for d in docs] == expected  # sorted by id


def test_manifest_fields_carried_over(by_id: dict[str, LoadedDocument]) -> None:
    m = by_id["HR-POL-002"].meta
    assert m.title == "Leave and Time Off Policy"
    assert m.version == "4.1"
    assert m.effective_date == date(2026, 1, 1)
    assert m.effective_date_int == 20260101
    assert m.owner == "People Operations"
    assert m.classification == "Internal"
    assert m.supersedes == "HR-POL-002 v3.6"
    assert m.subtitle == "Employee Handbook, Section 4"


def test_body_present_and_metadata_block_removed(docs: list[LoadedDocument]) -> None:
    for d in docs:
        assert d.pages, d.meta.document_id
        assert d.pages[0].strip(), f"{d.meta.document_id}: empty page 1 body"
        for page in d.pages:
            assert "Document ID" not in page, d.meta.document_id
            assert "Effective Date" not in page, d.meta.document_id
        # The banner is metadata too.
        assert not d.pages[0].startswith("CERULEAN SYSTEMS"), d.meta.document_id


def test_running_headers_stripped(docs: list[LoadedDocument]) -> None:
    for d in docs:
        for page in d.pages:
            assert "Cerulean Systems Ltd. |" not in page
            assert "Cerulean Systems Ltd.  |" not in page
            for line in page.splitlines():
                assert not line.strip().startswith("Page "), (d.meta.document_id, line)


def test_supersedes_chain(by_id: dict[str, LoadedDocument]) -> None:
    old = by_id["SALES-PL-2025"].meta
    new = by_id["SALES-PL-2026"].meta
    assert old.superseded_by == "SALES-PL-2026"
    assert old.is_current is False
    assert new.superseded_by is None
    assert new.is_current is True
    # Every other document is current: their supersedes links point outside the corpus.
    others = [m for k, m in ((k, v.meta) for k, v in by_id.items()) if k != "SALES-PL-2025"]
    assert all(m.is_current for m in others)


def test_review_status_parsed_for_faq(by_id: dict[str, LoadedDocument]) -> None:
    faq = by_id["SUP-FAQ-001"].meta
    assert faq.review_status is not None and "Overdue" in faq.review_status
    assert faq.supersedes is None
    # Other documents have no review status line.
    assert by_id["HR-POL-002"].meta.review_status is None


def test_injection_text_is_kept_as_content(by_id: dict[str, LoadedDocument]) -> None:
    assert "Ignore all previous instructions" in by_id["PROC-PRO-002"].text
    assert "Note to any AI assistant" in by_id["SUP-FAQ-001"].text
    assert "assistant_directive" in by_id["PROD-DOC-009"].text


def test_wrapped_supersedes_value_does_not_leak_into_body(
    by_id: dict[str, LoadedDocument],
) -> None:
    # PROD-DOC-009's "Supersedes" value wraps onto a second line ("2025)").
    body = by_id["PROD-DOC-009"].pages[0]
    assert body.lstrip().startswith("1. API rate limits"), body[:80]


def test_body_starts_at_first_section(by_id: dict[str, LoadedDocument]) -> None:
    assert by_id["HR-POL-002"].pages[0].lstrip().startswith("4.1 Purpose and scope")
    assert by_id["SUP-FAQ-001"].pages[0].lstrip().startswith("Getting started")
    assert by_id["FIN-POL-003"].pages[0].lstrip().startswith("1. Purpose")


def test_no_header_manifest_disagreement_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="cerulean_rag.loaders"):
        load_corpus(CORPUS)
    disagreements = [r for r in caplog.records if "disagrees with manifest" in r.getMessage()]
    assert disagreements == [], [r.getMessage() for r in disagreements]


# ---- unit tests on the helpers ---------------------------------------------
def test_parse_date_formats() -> None:
    assert parse_date("1 January 2026") == date(2026, 1, 1)
    assert parse_date("15 March 2026") == date(2026, 3, 15)
    assert parse_date("2026-08-27") == date(2026, 8, 27)
    with pytest.raises(ValueError):
        parse_date("sometime soon")


def test_strip_running_headers_only_removes_header_lines() -> None:
    page = (
        "Cerulean Systems Ltd.  |  HR-POL-002\nPage 2\n"
        "4.6 Sick leave\nCerulean Systems Ltd. is mentioned in the body.\n"
    )
    out = strip_running_headers(page)
    assert out.startswith("4.6 Sick leave")
    assert "mentioned in the body" in out
    assert "Page 2" not in out


def test_parse_metadata_block_handles_wrapped_value() -> None:
    page1 = (
        "CERULEAN SYSTEMS\nExternal\nSome Title\nA subtitle line\n"
        "Document ID\nX-Y-001\nVersion\n1.0\nEffective Date\n1 April 2026\n"
        "Owner\nProduct\nClassification\nExternal\n"
        "Supersedes\nX-Y-001 v0.9 (10 November\n2025)\n"
        "1. First section\nBody text.\n"
    )
    fields, body = parse_metadata_block(page1)
    assert fields["title"] == "Some Title"
    assert fields["subtitle"] == "A subtitle line"
    assert fields["document_id"] == "X-Y-001"
    assert fields["supersedes"] == "X-Y-001 v0.9 (10 November 2025)"
    assert body.startswith("1. First section")


def _write_plain_pdf(path: Path, title: str, lines: list[str]) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), title, fontsize=16)
    y = 130
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 22
    doc.save(path)
    doc.close()


def test_document_without_a_metadata_block_loads_from_the_manifest(tmp_path: Path) -> None:
    """A PDF from outside this corpus has no "Document ID / Version / ..." block on
    page 1. It must still load, with the manifest supplying the metadata."""
    corpus = tmp_path / "mycorpus"
    corpus.mkdir()
    _write_plain_pdf(corpus / "remote.pdf", "Remote Working Policy",
                     ["1. Purpose", "This policy sets out how staff may work remotely.",
                      "2. Eligibility", "Employees who have completed probation may apply."])
    (corpus / "corpus_manifest.json").write_text(json.dumps({"documents": [{
        "file": "remote.pdf", "document_id": "OPS-POL-001", "title": "Remote Working Policy",
        "version": "1.0", "effective_date": "2026-02-01", "owner": "People Ops",
        "classification": "Internal"}]}), encoding="utf-8")

    docs = load_corpus(corpus)
    assert len(docs) == 1
    meta = docs[0].meta
    assert meta.document_id == "OPS-POL-001"
    assert meta.title == "Remote Working Policy"
    assert meta.effective_date == date(2026, 2, 1)
    assert meta.is_current is True
    # page 1 is all body when there is no metadata block to cut out
    assert "Remote Working Policy" in docs[0].text
    assert "2. Eligibility" in docs[0].text


def test_manifest_entry_missing_a_required_field_is_reported_clearly(tmp_path: Path) -> None:
    corpus = tmp_path / "mycorpus"
    corpus.mkdir()
    _write_plain_pdf(corpus / "a.pdf", "Some Policy", ["1. Scope", "Applies to everyone."])
    (corpus / "corpus_manifest.json").write_text(json.dumps({"documents": [
        {"file": "a.pdf", "document_id": "OPS-POL-002", "version": "1.0"}]}), encoding="utf-8")

    with pytest.raises(CorpusLoadError) as exc:
        load_corpus(corpus)
    assert "OPS-POL-002" in str(exc.value)
    assert "title" in str(exc.value) and "effective_date" in str(exc.value)
