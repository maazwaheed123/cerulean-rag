"""Section-aware chunking with context headers.

One chunk is one logical section, tables included, or one FAQ question with its
answer; anything with neither falls back to a character splitter. Each chunk is
embedded behind a header such as

    [HR-POL-002 v4.1 | Leave and Time Off Policy | effective 2026-01-01 | current | 4.2 Annual leave entitlement]

so identity, dates and supersession status travel with the passage.

python -m cerulean_rag.chunking corpus prints every chunk.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from cerulean_rag.models import Chunk, DocumentMeta, LoadedDocument

log = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 1800          # a section longer than this is split ...
SPLIT_CHUNK_SIZE = 1500         # ... into parts of about this size ...
SPLIT_OVERLAP = 200             # ... with this much overlap
FALLBACK_CHUNK_SIZE = 900       # for documents with no detectable structure
FALLBACK_OVERLAP = 120
MIN_HEADINGS_FOR_SECTION_MODE = 2
MIN_QUESTIONS_FOR_FAQ_MODE = 3

# "4.2 Annual leave entitlement", "3. Notice periods", "7. Annex B — ..."
_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.{1,80})$")
# Heading text never reads like a sentence.
_SENTENCE_LIKE_RE = re.compile(r"[.:;,]$|\. ")
_QUESTION_RE = re.compile(r"\?\s*$")
_TABLE_ROW_RE = re.compile(r"^\|.*\|$")
MAX_GROUP_HEADING_CHARS = 40


@dataclass
class _Line:
    text: str
    page: int


@dataclass
class _Section:
    number: str | None      # "4.2", "3", or None for FAQ / fallback
    label: str              # "4.2 Annual leave entitlement" or "Plans and billing: Do you offer refunds?"
    slug: str               # id-safe form used in chunk_id
    lines: list[_Line]

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines).strip()

    @property
    def page(self) -> int:
        return self.lines[0].page if self.lines else 1


def _parse_number(num: str) -> tuple[int, ...]:
    return tuple(int(p) for p in num.split("."))


def _is_successor(prev: tuple[int, ...] | None, cur: tuple[int, ...]) -> bool:
    """Accepts the next sibling (4.2 -> 4.3), the first child (4 -> 4.1), or the
    next sibling of an ancestor (4.9 -> 5), and nothing else. That is what keeps
    a numbered list item inside section 4 from being read as a heading.
    """
    if prev is None:
        return True
    if cur == prev + (1,):
        return True
    for depth in range(len(prev), 0, -1):
        if len(cur) == depth and cur[:-1] == prev[: depth - 1] and cur[-1] == prev[depth - 1] + 1:
            return True
    return False


def match_heading(line: str, prev_number: tuple[int, ...] | None) -> tuple[str, str] | None:
    """(number, title) if the line is a heading that follows prev_number."""
    m = _HEADING_RE.match(line.strip())
    if not m:
        return None
    number, title = m.group(1), m.group(2).strip()
    if _SENTENCE_LIKE_RE.search(title):
        return None
    if not _is_successor(prev_number, _parse_number(number)):
        return None
    return number, title


def _document_lines(doc: LoadedDocument) -> list[_Line]:
    lines: list[_Line] = []
    for page_no, page in enumerate(doc.pages, start=1):
        for raw in page.splitlines():
            text = raw.rstrip()
            if text.strip():
                lines.append(_Line(text=text, page=page_no))
    return lines


def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit].rstrip("-") or "section"


def split_numbered_sections(lines: list[_Line]) -> list[_Section]:
    """Text before the first heading becomes a preamble."""
    sections: list[_Section] = []
    current = _Section(number="0", label="Preamble", slug="preamble", lines=[])
    prev: tuple[int, ...] | None = None
    for ln in lines:
        hit = match_heading(ln.text, prev)
        if hit:
            number, title = hit
            if current.lines:
                sections.append(current)
            current = _Section(number=number, label=ln.text.strip(), slug=number, lines=[ln])
            prev = _parse_number(number)
        else:
            current.lines.append(ln)
    if current.lines:
        sections.append(current)
    return sections


def split_faq_sections(lines: list[_Line]) -> list[_Section]:
    """Split on question lines; a short unpunctuated line before one is a group heading."""
    sections: list[_Section] = []
    group: str | None = None
    current: _Section | None = None
    for i, ln in enumerate(lines):
        text = ln.text.strip()
        next_text = lines[i + 1].text.strip() if i + 1 < len(lines) else ""
        is_group = (
            len(text) <= MAX_GROUP_HEADING_CHARS
            and not _QUESTION_RE.search(text)
            and not re.search(r"[.:;,\]]$", text)
            and _QUESTION_RE.search(next_text) is not None
        )
        if is_group:
            group = text
            continue
        if _QUESTION_RE.search(text):
            if current is not None:
                sections.append(current)
            label = f"{group}: {text}" if group else text
            current = _Section(number=None, label=label, slug=f"faq-{_slug(text)}", lines=[ln])
            continue
        if current is None:
            current = _Section(number=None, label=group or "Introduction", slug="intro", lines=[])
        current.lines.append(ln)
    if current is not None and current.lines:
        sections.append(current)
    return sections


def _count_questions(lines: list[_Line]) -> int:
    return sum(1 for ln in lines if _QUESTION_RE.search(ln.text))


def _count_headings(lines: list[_Line]) -> int:
    prev: tuple[int, ...] | None = None
    n = 0
    for ln in lines:
        hit = match_heading(ln.text, prev)
        if hit:
            n += 1
            prev = _parse_number(hit[0])
    return n


def context_header(meta: DocumentMeta, section_label: str) -> str:
    status = "current" if meta.is_current else f"superseded by {meta.superseded_by}"
    if meta.review_status:
        status += f"; review status: {meta.review_status}"
    return (
        f"[{meta.document_id} v{meta.version} | {meta.title} | "
        f"effective {meta.effective_date.isoformat()} | {status} | {section_label}]"
    )


def _make_chunk(meta: DocumentMeta, section: _Section, text: str, part: int, index: int) -> Chunk:
    label = section.label if part == 1 else f"{section.label} (part {part})"
    return Chunk(
        chunk_id=f"{meta.document_id}::{section.slug}::{part}",
        document_id=meta.document_id,
        section_number=section.number,
        section_label=label,
        part=part,
        page=section.page,
        chunk_index=index,
        text=text,
        text_for_embedding=f"{context_header(meta, label)}\n{text}",
        meta=meta,
    )


def _split_oversized(section: _Section) -> list[str]:
    """The heading line is repeated in every part."""
    heading = section.lines[0].text.strip() if section.number else section.label
    body = "\n".join(ln.text for ln in section.lines[1:]) if section.number else section.text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHUNK_SIZE - len(heading) - 1,
        chunk_overlap=SPLIT_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parts = splitter.split_text(body)
    return [f"{heading}\n{p.strip()}" for p in parts if p.strip()]


def chunk_document(doc: LoadedDocument, start_index: int = 0) -> list[Chunk]:
    """start_index seeds chunk_index so numbering is unique corpus-wide."""
    meta = doc.meta
    lines = _document_lines(doc)
    if not lines:
        log.warning("%s: no text to chunk", meta.document_id)
        return []

    if _count_headings(lines) >= MIN_HEADINGS_FOR_SECTION_MODE:
        sections = split_numbered_sections(lines)
        mode = "sections"
    elif _count_questions(lines) >= MIN_QUESTIONS_FOR_FAQ_MODE:
        sections = split_faq_sections(lines)
        mode = "faq"
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=FALLBACK_CHUNK_SIZE, chunk_overlap=FALLBACK_OVERLAP
        )
        sections = [
            _Section(number=None, label=f"Part {i + 1}", slug=f"part-{i + 1}", lines=[_Line(t, 1)])
            for i, t in enumerate(splitter.split_text(doc.text))
        ]
        mode = "fallback"

    chunks: list[Chunk] = []
    index = start_index
    for section in sections:
        text = section.text
        if not text:
            continue
        if len(text) <= MAX_CHUNK_CHARS:
            chunks.append(_make_chunk(meta, section, text, part=1, index=index))
            index += 1
            continue
        for part_no, part_text in enumerate(_split_oversized(section), start=1):
            chunks.append(_make_chunk(meta, section, part_text, part=part_no, index=index))
            index += 1

    log.debug("%s: %d chunks (%s mode)", meta.document_id, len(chunks), mode)
    return chunks


def chunk_documents(docs: list[LoadedDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, start_index=len(chunks)))
    log.info("chunked %d documents into %d chunks", len(docs), len(chunks))
    return chunks


def looks_like_table(text: str) -> bool:
    return sum(1 for ln in text.splitlines() if _TABLE_ROW_RE.match(ln.strip())) >= 3


def _main(argv: list[str]) -> int:
    from rich.console import Console
    from rich.table import Table

    from cerulean_rag.loaders import load_corpus

    corpus_dir = Path(argv[1]) if len(argv) > 1 else Path("corpus")
    logging.basicConfig(level=logging.WARNING)
    chunks = chunk_documents(load_corpus(corpus_dir))

    table = Table(title=f"{len(chunks)} chunks from {corpus_dir}", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("doc")
    table.add_column("section_label")
    table.add_column("chars", justify="right")
    table.add_column("tbl")
    table.add_column("first 60 chars")
    for c in chunks:
        first = c.text.splitlines()[0][:60] if c.text else ""
        table.add_row(
            str(c.chunk_index), c.document_id, c.section_label, str(c.char_len),
            "yes" if looks_like_table(c.text) else "", first,
        )
    console = Console(width=160)
    console.print(table)
    lengths = [c.char_len for c in chunks]
    console.print(
        f"chunks={len(chunks)} min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} "
        f"max={max(lengths)} tables={sum(looks_like_table(c.text) for c in chunks)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
