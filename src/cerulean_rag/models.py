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
