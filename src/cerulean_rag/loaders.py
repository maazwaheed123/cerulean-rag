"""PDF loading: page text, running-header removal, metadata parsing, manifest merge.

Layout of every corpus PDF as extracted by PyMuPDF (``page.get_text("text")``):

    Cerulean Systems Ltd.  |  <DOC-ID>        <- running header, line 1
    Page <n>                                   <- running header, line 2
    CERULEAN SYSTEMS                           <- banner (page 1 only)
    Internal | External                        <- banner classification
    <Title>
    <Subtitle>                                 <- optional italic line
    Document ID / <value>                      <- metadata block, one label per
    Version / <value>                             line followed by its value on
    Effective Date / <value>                      the next line(s)
    Owner / <value>
    Classification / <value>
    Supersedes / <value>   or   Review status / <value>
    <body ...>

The manifest is the authority for metadata. The PDF header is parsed as a
cross-check and any disagreement is logged as a warning. ``superseded_by`` and
``is_current`` are derived here from the supersedes chain across the corpus.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path

import pymupdf

from cerulean_rag.models import DocumentMeta, LoadedDocument

log = logging.getLogger(__name__)

MANIFEST_NAME = "corpus_manifest.json"

# Running header: "Cerulean Systems Ltd.  |  HR-POL-002" followed by "Page 1".
_HEADER_LINE_RE = re.compile(r"^Cerulean Systems Ltd\.\s*\|\s*\S+\s*$")
_PAGE_LINE_RE = re.compile(r"^Page\s+\d+\s*$")
# Banner on page 1.
_BANNER_RE = re.compile(r"^CERULEAN SYSTEMS\s*$")
_BANNER_CLASS_RE = re.compile(r"^(Internal|External|Public|Confidential|Restricted)\s*$")

# Labels of the metadata block, in the order they appear in the PDFs.
_META_LABELS: dict[str, str] = {
    "Document ID": "document_id",
    "Version": "version",
    "Effective Date": "effective_date",
    "Owner": "owner",
    "Classification": "classification",
    "Supersedes": "supersedes",
    "Review status": "review_status",
}
_LABEL_RE = re.compile(
    "^(" + "|".join(re.escape(k) for k in _META_LABELS) + r")\s*$", re.IGNORECASE
)

# "HR-POL-002 v3.6 (1 January 2024)" -> "HR-POL-002 v3.6"
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
# First token of a supersedes string, e.g. "SALES-PL-2025" from "SALES-PL-2025 v1.0"
_DOC_ID_RE = re.compile(r"^([A-Z]+-[A-Z]+-\d+)\b")


class CorpusLoadError(RuntimeError):
    """Raised when the corpus directory or manifest cannot be used."""


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def load_manifest(corpus_dir: str | Path) -> list[dict]:
    """Return the ``documents`` list from ``corpus_manifest.json``."""
    path = Path(corpus_dir) / MANIFEST_NAME
    if not path.is_file():
        raise CorpusLoadError(f"Manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusLoadError(f"Manifest is not valid JSON: {path}: {exc}") from exc
    docs = data.get("documents")
    if not isinstance(docs, list) or not docs:
        raise CorpusLoadError(f"Manifest has no 'documents' list: {path}")
    return docs


# --------------------------------------------------------------------------- #
# Page text
# --------------------------------------------------------------------------- #
def extract_pages(pdf_path: str | Path, render_tables: bool = True) -> list[str]:
    """Text of every page in reading order, with tables rendered as pipe rows.

    Plain ``get_text("text")`` emits each table cell on its own line, which
    loses the row association ("Atlas Professional" / "SAR 5,200" / ...). With
    ``render_tables`` the page's text blocks are taken in reading order, blocks
    that fall inside a detected table are replaced by the table rendered as
    ``| cell | cell |`` rows at the table's position, and everything else is
    left exactly as PyMuPDF extracted it.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise CorpusLoadError(f"PDF not found: {path}")
    with pymupdf.open(path) as doc:
        if not render_tables:
            return [page.get_text("text") for page in doc]
        return [_page_text_with_tables(page) for page in doc]


def render_table_rows(rows: list[list[str | None]]) -> str:
    """Render extracted table rows as Markdown-style pipe rows (header row first)."""
    def clean(cell: str | None) -> str:
        return re.sub(r"\s+", " ", (cell or "")).strip()

    lines: list[str] = []
    for i, row in enumerate(rows):
        cells = [clean(c) for c in row]
        if not any(cells):
            continue
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "|".join(" --- " for _ in cells) + "|")
    return "\n".join(lines)


def _page_text_with_tables(page: pymupdf.Page) -> str:
    tables = page.find_tables().tables
    if not tables:
        return page.get_text("text")

    table_rects = [pymupdf.Rect(t.bbox) for t in tables]
    items: list[tuple[float, float, str]] = []  # (y0, x0, text)

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, _block_no, block_type = block
        if block_type != 0 or not text.strip():
            continue
        centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(rect.contains(centre) for rect in table_rects):
            continue  # this text belongs to a table; rendered below
        items.append((y0, x0, text.rstrip("\n")))

    for table, rect in zip(tables, table_rects):
        rendered = render_table_rows(table.extract())
        if rendered:
            items.append((rect.y0, rect.x0, rendered))

    items.sort(key=lambda it: (round(it[0], 1), it[1]))
    return "\n".join(text for _, _, text in items) + "\n"


def strip_running_headers(page_text: str) -> str:
    """Remove the two-line running header and normalise whitespace.

    Only whole lines matching the header patterns are dropped, so body text
    that merely mentions the company name is untouched.
    """
    out: list[str] = []
    lines = page_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _HEADER_LINE_RE.match(line.strip()):
            i += 1
            # The page number sits on its own line directly after the header.
            if i < len(lines) and _PAGE_LINE_RE.match(lines[i].strip()):
                i += 1
            continue
        out.append(line.rstrip())
        i += 1
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def parse_date(value: str) -> date:
    """Parse ``1 January 2026`` or ISO ``2026-01-01``; raise ValueError otherwise."""
    v = value.strip()
    try:
        return date.fromisoformat(v)
    except ValueError:
        pass
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {value!r}")


# --------------------------------------------------------------------------- #
# Metadata block on page 1
# --------------------------------------------------------------------------- #
def parse_metadata_block(page1_text: str) -> tuple[dict[str, str], str]:
    """Split page 1 (headers already stripped) into (header_fields, body_text).

    ``header_fields`` holds ``title``, ``subtitle`` (may be absent) and one entry
    per metadata label found. The banner and the metadata block are removed
    from the returned body so they never become chunks.
    """
    lines = [ln.strip() for ln in page1_text.splitlines()]
    # Drop leading blanks.
    while lines and not lines[0]:
        lines.pop(0)

    fields: dict[str, str] = {}
    idx = 0
    if idx < len(lines) and _BANNER_RE.match(lines[idx]):
        idx += 1
        if idx < len(lines) and _BANNER_CLASS_RE.match(lines[idx]):
            idx += 1

    # Find the first metadata label; the lines between banner and label are
    # the title and (optionally) subtitle.
    label_positions = [i for i in range(idx, len(lines)) if _LABEL_RE.match(lines[i])]
    if not label_positions:
        raise CorpusLoadError("No metadata block ('Document ID' ...) found on page 1")
    first_label = label_positions[0]
    heading = [ln for ln in lines[idx:first_label] if ln]
    if heading:
        fields["title"] = heading[0]
    if len(heading) > 1:
        fields["subtitle"] = " ".join(heading[1:])

    # Walk label/value pairs. A value normally occupies one line; it continues
    # onto following lines while a parenthesis opened in it is still unclosed
    # (PROD-DOC-009's "Supersedes" wraps like this).
    i = first_label
    body_start = first_label
    while i < len(lines):
        m = _LABEL_RE.match(lines[i])
        if not m:
            break
        key = _META_LABELS[_canonical_label(m.group(1))]
        i += 1
        value_parts: list[str] = []
        while i < len(lines):
            if _LABEL_RE.match(lines[i]):
                break
            if value_parts and not _has_unclosed_paren(" ".join(value_parts)):
                break
            if lines[i]:
                value_parts.append(lines[i])
            i += 1
        fields[key] = " ".join(value_parts).strip()
        body_start = i

    body = "\n".join(ln for ln in lines[body_start:]).strip("\n")
    return fields, body


def _canonical_label(label: str) -> str:
    for known in _META_LABELS:
        if known.lower() == label.strip().lower():
            return known
    raise KeyError(label)


def _has_unclosed_paren(text: str) -> bool:
    return text.count("(") > text.count(")")


# --------------------------------------------------------------------------- #
# Merge + derive
# --------------------------------------------------------------------------- #
def _normalise_supersedes(value: str | None) -> str | None:
    if value is None:
        return None
    v = _TRAILING_PAREN_RE.sub("", value).strip()
    return v or None


def _compare_header_with_manifest(doc_id: str, header: dict[str, str], entry: dict) -> None:
    """Log a warning for every field where the PDF header disagrees with the manifest."""
    checks: list[tuple[str, str | None, str | None]] = [
        ("document_id", header.get("document_id"), entry.get("document_id")),
        ("version", header.get("version"), entry.get("version")),
        ("owner", header.get("owner"), entry.get("owner")),
        ("classification", header.get("classification"), entry.get("classification")),
        ("title", header.get("title"), entry.get("title")),
        (
            "supersedes",
            _normalise_supersedes(header.get("supersedes")),
            _normalise_supersedes(entry.get("supersedes")),
        ),
    ]
    eff = header.get("effective_date")
    if eff is not None:
        try:
            checks.append(("effective_date", parse_date(eff).isoformat(), entry.get("effective_date")))
        except ValueError:
            log.warning("%s: PDF effective date %r is not parseable", doc_id, eff)
    for name, pdf_value, manifest_value in checks:
        if pdf_value is None:
            continue
        if _soft_eq(pdf_value, manifest_value):
            continue
        log.warning(
            "%s: PDF header %s=%r disagrees with manifest %r (manifest used)",
            doc_id, name, pdf_value, manifest_value,
        )


def _soft_eq(a: str | None, b: str | None) -> bool:
    """Case-insensitive comparison tolerant of punctuation-only differences.

    The manifest writes titles without the em dash the PDFs use
    ("Atlas Platform — Customer FAQ" vs "Atlas Platform Customer FAQ"), so a
    spaced dash of any kind is treated as a plain space.
    """
    def norm(s: str | None) -> str:
        if s is None:
            return ""
        s = re.sub(r"\s+[—–-]\s+", " ", s)  # " — ", " – ", " - " -> " "
        return re.sub(r"\s+", " ", s).strip().lower()
    return norm(a) == norm(b)


def derive_supersedes_chain(metas: list[DocumentMeta]) -> None:
    """Set ``superseded_by`` / ``is_current`` in place from the ``supersedes`` links.

    Document A is superseded when some document B's ``supersedes`` value names
    A's id as its first token (e.g. ``"SALES-PL-2025 v1.0"``). Links to versions
    not in the corpus leave the target untouched.
    """
    by_id = {m.document_id: m for m in metas}
    for m in metas:
        m.superseded_by = None
        m.is_current = True
    for newer in metas:
        if not newer.supersedes:
            continue
        match = _DOC_ID_RE.match(newer.supersedes.strip())
        if not match:
            continue
        older = by_id.get(match.group(1))
        if older is None or older is newer:
            continue
        if older.superseded_by and older.superseded_by != newer.document_id:
            log.warning(
                "%s is superseded by both %s and %s; keeping the later effective date",
                older.document_id, older.superseded_by, newer.document_id,
            )
            if by_id[older.superseded_by].effective_date >= newer.effective_date:
                continue
        older.superseded_by = newer.document_id
        older.is_current = False


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_document(corpus_dir: str | Path, entry: dict) -> LoadedDocument:
    """Load one manifest entry: extract, clean, parse header, build metadata."""
    corpus_dir = Path(corpus_dir)
    doc_id = entry.get("document_id", "<unknown>")
    pdf_path = corpus_dir / entry["file"]

    raw_pages = extract_pages(pdf_path)
    if not raw_pages or not raw_pages[0].strip():
        raise CorpusLoadError(f"{doc_id}: no extractable text in {pdf_path.name}")

    cleaned = [strip_running_headers(p) for p in raw_pages]
    header, body1 = parse_metadata_block(cleaned[0])
    cleaned[0] = body1
    _compare_header_with_manifest(doc_id, header, entry)

    try:
        effective = parse_date(str(entry["effective_date"]))
    except (KeyError, ValueError) as exc:
        raise CorpusLoadError(f"{doc_id}: bad effective_date in manifest: {exc}") from exc

    meta = DocumentMeta(
        document_id=entry["document_id"],
        file=entry["file"],
        title=entry.get("title") or header.get("title", ""),
        version=str(entry.get("version") or header.get("version", "")),
        effective_date=effective,
        owner=entry.get("owner") or header.get("owner", ""),
        classification=entry.get("classification") or header.get("classification", ""),
        supersedes=_normalise_supersedes(entry.get("supersedes")),
        review_status=header.get("review_status") or None,
        subtitle=header.get("subtitle") or None,
    )
    return LoadedDocument(meta=meta, pages=cleaned)


def load_corpus(corpus_dir: str | Path) -> list[LoadedDocument]:
    """Load every document listed in the manifest, derive the supersedes chain,
    and return the documents sorted by ``document_id``."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise CorpusLoadError(f"Corpus directory not found: {corpus_dir}")

    docs: list[LoadedDocument] = []
    for entry in load_manifest(corpus_dir):
        doc = load_document(corpus_dir, entry)
        docs.append(doc)
        log.debug(
            "loaded %s v%s (%d pages, %d chars)",
            doc.meta.document_id, doc.meta.version, len(doc.pages), len(doc.text),
        )

    derive_supersedes_chain([d.meta for d in docs])
    docs.sort(key=lambda d: d.meta.document_id)

    superseded = [d.meta.document_id for d in docs if not d.meta.is_current]
    log.info(
        "loaded %d documents from %s (%d superseded: %s)",
        len(docs), corpus_dir, len(superseded), ", ".join(superseded) or "none",
    )
    return docs
