"""Shared data types.

Step 2 defines the document-level types. Chunk, retrieval and answer types are
added in the steps that introduce them.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, computed_field, field_validator


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
    injection_payloads: list[str] = Field(default_factory=list)
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
            "injection_payloads": json.dumps(self.injection_payloads, ensure_ascii=False),
        }


# --------------------------------------------------------------------------- #
# LLM output schema (Part 5.3)
# --------------------------------------------------------------------------- #
DECISIONS: tuple[str, ...] = (
    "answer",
    "insufficient_evidence",
    "needs_clarification",
    "conflict_resolved",
    "refused",
)


class Citation(BaseModel):
    """A reference to one excerpt that supports the answer."""

    document_id: str
    section: str = ""
    quote: str = Field(default="", description="short verbatim or near-verbatim supporting quote")

    @field_validator("document_id", mode="before")
    @classmethod
    def _strip_id(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v


class Conflict(BaseModel):
    """Two or more excerpts disagreeing on the same fact, and how it was resolved."""

    topic: str
    positions: list[str] = Field(default_factory=list)
    resolution: str = ""
    reasoning: str = ""


class AnswerSchema(BaseModel):
    """The single JSON object the generation model must return.

    ``decision`` is a plain string constrained by a validator rather than a
    ``Literal`` so that small models' casing or hyphenation variants
    ("Insufficient-Evidence") are normalised instead of failing the parse.
    """

    decision: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    clarification_options: list[str] = Field(default_factory=list)
    injection_noticed: bool = False
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("decision", mode="before")
    @classmethod
    def _normalise_decision(cls, v: object) -> object:
        if not isinstance(v, str):
            raise ValueError("decision must be a string")
        key = v.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {v!r}")
        return key


# --------------------------------------------------------------------------- #
# Retrieval results (Step 6)
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    """A chunk in the final fused result set with the evidence for its rank."""

    chunk: Chunk
    rrf_score: float
    vector_sim: float | None = None      # best cosine similarity across sub-queries, None if BM25-only
    bm25_score: float | None = None      # best BM25 score across sub-queries, None if vector-only
    sources: list[str] = Field(default_factory=list)   # e.g. ["vector", "bm25"]

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    def summary(self) -> dict:
        """Compact record for logs and eval output."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "section": self.chunk.section_label,
            "rrf": round(self.rrf_score, 4),
            "vector_sim": None if self.vector_sim is None else round(self.vector_sim, 3),
            "bm25": None if self.bm25_score is None else round(self.bm25_score, 2),
            "sources": self.sources,
        }


class RetrievalBundle(BaseModel):
    """Everything retrieval hands to prompt assembly, fully inspectable without an LLM."""

    question: str
    sub_queries: list[str] = Field(default_factory=list)   # extra queries derived from the question
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    best_sim: float | None = None
    signals: list[str] = Field(default_factory=list)       # trusted, system-generated
    metadata_notes: list[str] = Field(default_factory=list)  # trusted, from the manifest
    ambiguous: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def document_ids(self) -> list[str]:
        seen: list[str] = []
        for rc in self.chunks:
            if rc.document_id not in seen:
                seen.append(rc.document_id)
        return seen
