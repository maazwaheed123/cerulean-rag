"""Shared data types.

Step 2 defines the document-level types. Chunk, retrieval and answer types are
added in the steps that introduce them.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, computed_field


class DocumentMeta(BaseModel):
    """Metadata for one corpus document.

    Source of truth is ``corpus_manifest.json``; the PDF header block is parsed
    as a cross-check. ``superseded_by`` and ``is_current`` are derived from the
    supersedes chain across the corpus, never read from a file.
    """

    document_id: str
    file: str
    title: str
    version: str
    effective_date: date
    owner: str
    classification: str
    supersedes: str | None = None
    superseded_by: str | None = None
    is_current: bool = True
    review_status: str | None = None
    subtitle: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_date_int(self) -> int:
        """``yyyymmdd`` integer form, for numeric comparisons in Chroma filters."""
        d = self.effective_date
        return d.year * 10000 + d.month * 100 + d.day

    def short_label(self) -> str:
        """Human-readable one-liner used in context headers and citations."""
        return f"{self.document_id} v{self.version} ({self.title}, effective {self.effective_date.isoformat()})"


class LoadedDocument(BaseModel):
    """A corpus document after loading: metadata plus cleaned page texts.

    ``pages[i]`` is the text of page ``i+1`` with running headers removed; the
    metadata block has been cut out of page 1 because it is carried in ``meta``.
    """

    meta: DocumentMeta
    pages: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """All pages joined with a blank line between them."""
        return "\n\n".join(p for p in self.pages if p.strip())


class Chunk(BaseModel):
    """One retrievable unit: a logical section (or FAQ question) of a document.

    ``text`` is the raw section text for display and citation. ``text_for_embedding``
    is the same text with a one-line context header prepended so the embedding
    and the keyword index both carry document identity, dates and status.
    """

    chunk_id: str
    document_id: str
    section_number: str | None = None
    section_label: str
    part: int = 1
    page: int = 1
    chunk_index: int = 0
    text: str
    text_for_embedding: str
    has_injection: bool = False
    injection_spans: list[dict] = Field(default_factory=list)
    meta: DocumentMeta

    @property
    def char_len(self) -> int:
        return len(self.text)

    def to_metadata(self) -> dict[str, str | int | float | bool]:
        """Flat, scalar-only metadata for the vector store (no None values)."""
        import json

        m = self.meta
        return {
            "chunk_id": self.chunk_id,
            "document_id": m.document_id,
            "file": m.file,
            "title": m.title,
            "version": m.version,
            "effective_date": m.effective_date.isoformat(),
            "effective_date_int": m.effective_date_int,
            "owner": m.owner,
            "classification": m.classification,
            "supersedes": m.supersedes or "",
            "superseded_by": m.superseded_by or "",
            "is_current": m.is_current,
            "review_status": m.review_status or "",
            "subtitle": m.subtitle or "",
            "section_label": self.section_label,
            "section_number": self.section_number or "",
            "part": self.part,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "char_len": self.char_len,
            "has_injection": self.has_injection,
            "injection_spans": json.dumps(self.injection_spans, ensure_ascii=False),
        }
