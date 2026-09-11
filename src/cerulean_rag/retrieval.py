"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion.

Pipeline for one question (no LLM involved anywhere here):

1. ``split_subqueries``   compound questions ("summarise X and Y") become two
                          extra queries; the full question is always searched too.
2. per query              vector top-k (cosine similarity) and BM25 top-k.
3. ``rrf_fuse``           all result lists are fused with RRF (k=60), de-duplicated
                          by chunk id, trimmed to TOP_K. Each chunk keeps its best
                          vector similarity and BM25 score as evidence.
4. ``build_signals``      trusted text for the prompt: similarity band, and an
                          ambiguity warning for short questions whose hits spread
                          across several documents with near-equal scores.
5. ``build_metadata_notes`` trusted text from the manifest: status of each
                          retrieved document relative to AS_OF_DATE, supersedes
                          links, review status, chronology, and any precedence
                          clause found in the retrieved text.

Why BM25 as well as vectors: this corpus is full of exact tokens that dense
embeddings blur ("SAR 25,000", "Enterprise", "probation", document ids,
"14 calendar days"). Fusion lets either side rescue the other.
"""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from cerulean_rag.config import Settings, get_settings
from cerulean_rag.ingest import load_chunks_jsonl, open_vector_store
from cerulean_rag.models import Chunk, DocumentMeta, RetrievalBundle, RetrievedChunk

log = logging.getLogger(__name__)

RRF_K = 60
MAX_SUB_QUERIES = 3
GOOD_SIM_MARGIN = 0.10            # "good" band starts this far above SIM_THRESHOLD
AMBIGUITY_MAX_CONTENT_TOKENS = 2
AMBIGUITY_MIN_DOCS = 3
AMBIGUITY_MAX_SIM_SPREAD = 0.08

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset("""
a an and are as at be by for from has have how i in is it its of on or that the
this to was were what when where which who whom why will with you your do does
did can could should would may might must shall me my we our us they them their
there here about into over under than then so such if not no yes please tell give
""".split())

# Compound questions start like this; splitting a narrative sentence on "and"
# ("joins on 1 March and leaves on 15 September") would be wrong.
_COMPOUND_LEAD_RE = re.compile(
    r"^\s*(summari[sz]e|compare|contrast|explain|list|describe|outline|"
    r"what\s+are|tell\s+me\s+about|give\s+me)\b\s*(the\s+)?",
    re.I,
)
_SPLIT_RE = re.compile(r"\s*;\s*(?:and\s+(?:also\s+)?)?|\s+and\s+(?:also\s+)?", re.I)
_LEADING_THE_RE = re.compile(r"^\s*(the|our|its|their)\s+", re.I)
_PRECEDENCE_RE = re.compile(
    r"\b(prevails?|operative document|takes? precedence|governs|overrides?)\b", re.I
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Tokenisation + sub-queries
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stop words removed, numbers kept (BM25 and heuristics)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def content_tokens(question: str) -> list[str]:
    """Tokens that carry topic meaning (used by the ambiguity heuristic)."""
    return tokenize(question)


def split_subqueries(question: str) -> list[str]:
    """Return extra queries for a compound question, else an empty list.

    Splits only when the question opens with a summarise/compare/list-style
    lead and both halves have at least three alphabetic tokens.
    """
    lead = _COMPOUND_LEAD_RE.match(question)
    if not lead:
        return []
    body = question[lead.end():].strip().rstrip("?.!")
    parts = [p.strip() for p in _SPLIT_RE.split(body) if p and p.strip()]
    if len(parts) < 2:
        return []
    cleaned: list[str] = []
    for part in parts[:MAX_SUB_QUERIES]:
        part = _LEADING_THE_RE.sub("", part).strip()
        alpha = [t for t in _TOKEN_RE.findall(part.lower()) if t.isalpha()]
        if len(alpha) < 3:
            return []
        cleaned.append(part)
    return cleaned


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def rrf_fuse(
    ranked_lists: list[tuple[str, list[tuple[str, float]]]],
    top_k: int,
    k: int = RRF_K,
) -> list[tuple[str, float, dict[str, float], set[str]]]:
    """Reciprocal Rank Fusion.

    ``ranked_lists`` is ``[(source_name, [(chunk_id, score), ...]), ...]`` with each
    inner list already ranked best-first. Returns ``(chunk_id, rrf_score,
    best_scores_by_source, sources)`` for the top_k fused ids.
    """
    rrf: dict[str, float] = {}
    best: dict[str, dict[str, float]] = {}
    sources: dict[str, set[str]] = {}
    for source, results in ranked_lists:
        for rank, (chunk_id, score) in enumerate(results, start=1):
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(chunk_id, {})
            if score > best[chunk_id].get(source, float("-inf")):
                best[chunk_id][source] = score
            sources.setdefault(chunk_id, set()).add(source)
    order = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [(cid, s, best[cid], sources[cid]) for cid, s in order]


# --------------------------------------------------------------------------- #
# Signals + metadata notes (trusted prompt material)
# --------------------------------------------------------------------------- #
def similarity_band(best_sim: float | None, threshold: float) -> str:
    if best_sim is None:
        return "LOW — no vector matches; answer only if the excerpts state the fact explicitly"
    if best_sim >= threshold + GOOD_SIM_MARGIN:
        return "good"
    if best_sim >= threshold:
        return "adequate"
    return "LOW — answer only if the excerpts state the fact explicitly"


def is_ambiguous(question: str, chunks: list[RetrievedChunk]) -> bool:
    """Short question + hits spread over several documents with near-equal scores."""
    if len(content_tokens(question)) > AMBIGUITY_MAX_CONTENT_TOKENS:
        return False
    docs = {rc.document_id for rc in chunks}
    if len(docs) < AMBIGUITY_MIN_DOCS:
        return False
    sims = [rc.vector_sim for rc in chunks if rc.vector_sim is not None]
    if len(sims) >= 3:
        sims = sorted(sims, reverse=True)
        return (sims[0] - sims[2]) < AMBIGUITY_MAX_SIM_SPREAD
    return True


def build_signals(question: str, chunks: list[RetrievedChunk], best_sim: float | None,
                  threshold: float) -> tuple[list[str], bool]:
    signals: list[str] = []
    band = similarity_band(best_sim, threshold)
    if best_sim is None:
        signals.append(f"Best passage similarity: none ({band}).")
    else:
        signals.append(f"Best passage similarity: {best_sim:.2f} ({band}).")
    ambiguous = is_ambiguous(question, chunks)
    if ambiguous:
        docs = []
        for rc in chunks:
            if rc.document_id not in docs:
                docs.append(rc.document_id)
        signals.append(
            f"The question is short and the excerpts span {len(docs)} documents on different "
            f"topics ({', '.join(docs)}); it may be ambiguous — consider asking which is meant."
        )
    return signals, ambiguous


def _doc_label(m: DocumentMeta) -> str:
    return f"{m.document_id} (v{m.version}, effective {m.effective_date.isoformat()})"


def _precedence_sentences(text: str) -> list[str]:
    out = []
    flat = re.sub(r"\s+", " ", text)
    for sentence in _SENTENCE_SPLIT_RE.split(flat):
        if _PRECEDENCE_RE.search(sentence) and not sentence.startswith("|"):
            out.append(sentence.strip()[:240])
    return out


def build_metadata_notes(chunks: list[RetrievedChunk], as_of, all_meta: dict[str, DocumentMeta]) -> list[str]:
    """Trusted notes about the retrieved documents, derived from the manifest chain."""
    notes: list[str] = [f"Today is {as_of.isoformat()}."]
    metas: dict[str, DocumentMeta] = {}
    for rc in chunks:
        metas.setdefault(rc.document_id, rc.chunk.meta)
    if not metas:
        return notes

    # chronology, oldest first, so the model never has to order dates itself
    ordered = sorted(metas.values(), key=lambda m: (m.effective_date, m.document_id))
    chain = " < ".join(f"{m.document_id} ({m.effective_date.isoformat()})" for m in ordered)
    notes.append(f"Chronology of retrieved documents by effective date, oldest first: {chain}.")
    future = [m for m in ordered if m.effective_date > as_of]
    if future:
        notes.append(
            "Not yet in effect as of today: " + ", ".join(_doc_label(m) for m in future) + "."
        )

    for m in ordered:
        if m.is_current:
            line = f"{_doc_label(m)} is current."
        else:
            newer = all_meta.get(m.superseded_by or "")
            newer_label = _doc_label(newer) if newer else m.superseded_by
            line = f"{_doc_label(m)} is SUPERSEDED by {newer_label}; it is historical, not current."
        if m.review_status:
            line += (f" Review status: {m.review_status}; treat it as weaker evidence where "
                     "it conflicts with a current document.")
        notes.append(line)

    # explicit link lines when both ends of a supersedes relation are retrieved
    for m in ordered:
        if m.superseded_by and m.superseded_by in metas:
            newer = metas[m.superseded_by]
            notes.append(
                f"{_doc_label(newer)} SUPERSEDES {_doc_label(m)}. As of {as_of.isoformat()}, "
                f"{newer.document_id} is the current version; figures in {m.document_id} are historical."
            )
    # supersedes pointing outside the corpus
    for m in ordered:
        if m.supersedes and not any(m.supersedes.startswith(x) for x in all_meta):
            notes.append(f"{m.document_id} supersedes {m.supersedes}, which is not in the knowledge base.")

    # precedence clauses present in the retrieved text
    seen: set[str] = set()
    for rc in chunks:
        for sentence in _precedence_sentences(rc.chunk.text):
            key = (rc.document_id, sentence)
            if key in seen:
                continue
            seen.add(key)
            notes.append(
                f"{rc.document_id} §{rc.chunk.section_label} contains a precedence clause: \"{sentence}\""
            )
    return notes


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """Holds the chunk table, the BM25 index and the vector store for one process."""

    def __init__(self, settings: Settings | None = None, chunks: list[Chunk] | None = None,
                 store=None) -> None:
        self.settings = settings or get_settings()
        self.chunks: list[Chunk] = chunks if chunks is not None else load_chunks_jsonl(
            Path(self.settings.CHUNKS_FILE)
        )
        if not self.chunks:
            raise RuntimeError("no chunks available; run `python scripts/ingest.py`")
        self.by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self.all_meta: dict[str, DocumentMeta] = {}
        for c in self.chunks:
            self.all_meta.setdefault(c.document_id, c.meta)
        self._bm25 = BM25Okapi([tokenize(c.text_for_embedding) for c in self.chunks])
        self._store = store
        log.info("retriever ready: %d chunks, %d documents, bm25=%s",
                 len(self.chunks), len(self.all_meta), self.settings.USE_BM25)

    # -- individual searches -------------------------------------------------
    @property
    def store(self):
        if self._store is None:
            self._store = open_vector_store(self.settings)
        return self._store

    def vector_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity) best-first."""
        hits = self.store.similarity_search_with_score(query, k=k)
        out: list[tuple[str, float]] = []
        for doc, distance in hits:
            cid = doc.metadata.get("chunk_id")
            if cid in self.by_id:
                out.append((cid, 1.0 - float(distance)))
        return out

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """(chunk_id, bm25 score) best-first; zero-score chunks are dropped."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in ranked if scores[i] > 0]

    # -- full retrieval ------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalBundle:
        s = self.settings
        top_k = top_k or s.TOP_K
        k_each = s.RETRIEVE_K_PER_SOURCE
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        sub_queries = split_subqueries(question)
        queries = [question] + [q for q in sub_queries if q.lower() != question.lower()]

        ranked_lists: list[tuple[str, list[tuple[str, float]]]] = []
        t_vec = time.perf_counter()
        for q in queries:
            ranked_lists.append(("vector", self.vector_search(q, k_each)))
        timings["vector_ms"] = (time.perf_counter() - t_vec) * 1000
        if s.USE_BM25:
            t_bm = time.perf_counter()
            for q in queries:
                ranked_lists.append(("bm25", self.bm25_search(q, k_each)))
            timings["bm25_ms"] = (time.perf_counter() - t_bm) * 1000

        fused = rrf_fuse(ranked_lists, top_k=top_k)
        chunks = [
            RetrievedChunk(
                chunk=self.by_id[cid],
                rrf_score=score,
                vector_sim=best.get("vector"),
                bm25_score=best.get("bm25"),
                sources=sorted(srcs),
            )
            for cid, score, best, srcs in fused
        ]
        sims = [rc.vector_sim for rc in chunks if rc.vector_sim is not None]
        best_sim = max(sims) if sims else None

        signals, ambiguous = build_signals(question, chunks, best_sim, s.SIM_THRESHOLD)
        notes = build_metadata_notes(chunks, s.AS_OF_DATE, self.all_meta)
        timings["retrieval_total_ms"] = (time.perf_counter() - t0) * 1000

        log.info(
            "retrieved %d chunks from %s for %r (sub-queries=%d, best_sim=%s, ambiguous=%s)",
            len(chunks), ",".join(dict.fromkeys(rc.document_id for rc in chunks)), question[:60],
            len(sub_queries), f"{best_sim:.3f}" if best_sim is not None else None, ambiguous,
        )
        return RetrievalBundle(
            question=question, sub_queries=sub_queries, chunks=chunks, best_sim=best_sim,
            signals=signals, metadata_notes=notes, ambiguous=ambiguous, timings_ms=timings,
        )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide retriever built from the default settings (loads once)."""
    return Retriever(get_settings())
