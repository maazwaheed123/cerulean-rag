"""Deterministic security layer: three independent defences.

1. **Injection scanner** (ingest time). ``scan_text`` finds text inside a
   document that addresses an AI assistant or tries to issue commands, and
   ``extract_payloads`` pulls out the sentences and claims such text wants
   repeated. Flagged chunks are still indexed (the content is real company
   material) but are tagged, and their payloads are checked against every
   answer.
2. **Input guard** (before retrieval). ``check_user_input`` refuses prompt-
   extraction attempts and malformed input without spending an LLM call. It is
   deliberately narrow: policy-bypass requests are *not* blocked here, the
   model declines those under its own rules and the output checks make sure
   nothing leaked.
3. **Output checks** (after generation). ``verify_answer`` drops citations to
   documents that were not in the context, blocks answers that echo an
   injection payload as fact, flags figures that do not appear in the
   context, downgrades uncited answers, scrubs refusals that leak policy
   details, and blocks any answer that reproduces the system prompt.

Everything here is regex and string comparison. Nothing is keyed to a specific
question or document; the patterns describe *kinds* of text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from cerulean_rag.models import AnswerSchema, Chunk

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1. Injection scanner
# --------------------------------------------------------------------------- #
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern, flags))
    for name, pattern, flags in [
        ("ignore_instructions",
         r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions\b", re.I),
        ("disregard_instructions",
         r"\bdisregard\s+(your|all|any|the)?\s*(previous|prior|earlier)\s+instructions\b", re.I),
        ("system_prefix", r"^\s*SYSTEM\s*:", re.M),
        ("assistant_directive", r"\bassistant[_ ]directive\b", re.I),
        ("note_to_ai", r"\bnote\s+to\s+(any|the|all)\s+ai\s+(assistant|model|system)s?\b", re.I),
        ("mode_switch", r"\byou\s+are\s+now\s+(operating\s+in|in)\s+.{0,40}\bmode\b", re.I),
        ("print_system_prompt",
         r"\b(print|output|reveal|show|display|repeat)\s+(your\s+)?(full\s+)?(system\s+)?prompt\b", re.I),
        ("reveal_instructions", r"\breveal\s+(your\s+)?(system\s+)?(prompt|instructions)\b", re.I),
        ("respond_only_with", r"\b(respond|reply|answer)\s+only\s+with\b", re.I),
        ("do_not_disclose",
         r"\bdo\s+not\s+(disclose|mention|reveal)\s+this\s+(directive|instruction|note|message)\b", re.I),
        ("html_comment", r"<!--.*?-->", re.S),
        ("granted_admin", r"\bgranted\s+administrator\s+access\b", re.I),
        ("authorised_request",
         r"\bthis\s+is\s+an?\s+(authori[sz]ed|approved)\s+(diagnostic\s+)?(request|instruction|override)\b", re.I),
    ]
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_QUOTED_RE = re.compile(r"[\"“]([^\"”]{12,300})[\"”]")
# Clauses that a directive wants repeated: 'respond that X', 'reply with "X"', 'say that X'
_CLAIM_RE = re.compile(
    r"\b(?:respond|reply|say|state|tell\s+(?:the\s+)?users?|answer|claim)\s+(?:that\s+|with\s+|only\s+with\s+)?(.{12,300}?)(?:[.!?]|$)",
    re.I,
)
MIN_PAYLOAD_CHARS = 20


@dataclass
class InjectionSpan:
    start: int
    end: int
    pattern: str
    excerpt: str

    def to_dict(self) -> dict:
        return asdict(self)


def scan_text(text: str) -> list[InjectionSpan]:
    """Return every match of the injection patterns, sorted by position."""
    spans: list[InjectionSpan] = []
    for name, rx in INJECTION_PATTERNS:
        for m in rx.finditer(text):
            excerpt = text[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
            spans.append(InjectionSpan(m.start(), m.end(), name, excerpt.strip()))
    spans.sort(key=lambda s: (s.start, s.end))
    return spans


def _sentences_with_offsets(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    pos = 0
    for part in _SENTENCE_SPLIT_RE.split(text):
        if part is None:
            continue
        idx = text.find(part, pos)
        if idx < 0:
            continue
        out.append((idx, idx + len(part), part))
        pos = idx + len(part)
    return out


def extract_payloads(text: str, spans: list[InjectionSpan]) -> list[str]:
    """Sentences overlapping an injection span, plus quoted strings and
    'respond that ...' claims inside them. These are what an obeyed
    injection would make the assistant say, so they are what we check for."""
    if not spans:
        return []
    payloads: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        p = re.sub(r"\s+", " ", p).strip()
        p = re.sub(r"^<!--\s*|\s*-->$", "", p)
        p = p.strip().strip("[]\"“”'").strip().rstrip(".!?").strip()
        if len(p) >= MIN_PAYLOAD_CHARS and p.lower() not in seen:
            seen.add(p.lower())
            payloads.append(p)

    # Sentences that overlap a span, individually ...
    region_parts: list[str] = []
    for s_start, s_end, sentence in _sentences_with_offsets(text):
        if any(sp.start < s_end and sp.end > s_start for sp in spans):
            add(sentence)
            region_parts.append(sentence)

    # ... and the merged region, whitespace-normalised, for quotes and claims
    # that the sentence splitter may have cut (a period inside a closing quote,
    # or a line break inside a clause).
    region = re.sub(r"\s+", " ", " ".join(region_parts))
    for q in _QUOTED_RE.findall(region):
        add(q)
    for claim in _CLAIM_RE.findall(region):
        for clause in re.split(r"\s+and\s+that\s+|;\s*", claim):
            add(clause)
    return payloads


def annotate_chunks(chunks: list[Chunk]) -> int:
    """Set ``has_injection``, ``injection_spans`` and ``injection_payloads`` on each chunk in place.
    Returns the number of chunks flagged."""
    flagged = 0
    for chunk in chunks:
        spans = scan_text(chunk.text)
        chunk.has_injection = bool(spans)
        chunk.injection_spans = [s.to_dict() for s in spans]
        chunk.injection_payloads = extract_payloads(chunk.text, spans)
        if spans:
            flagged += 1
            log.warning(
                "injection patterns in %s (%s): %s",
                chunk.chunk_id, chunk.section_label, ", ".join(sorted({s.pattern for s in spans})),
            )
    return flagged


# --------------------------------------------------------------------------- #
# 2. Input guard
# --------------------------------------------------------------------------- #
MAX_QUESTION_CHARS = 2000

EXTRACTION_REFUSAL = (
    "I can't share my configuration or instructions, but I'm happy to answer "
    "questions about the Cerulean Systems documents."
)

_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in [
        # "repeat the instructions you were given", "print your system prompt", "show me your rules"
        r"\b(repeat|print|show|reveal|display|output|give|tell|share|dump|paste|recite)\b.{0,30}\b"
        r"(your|the)\s+(system\s+prompt|initial\s+prompt|hidden\s+prompt|prompt|configuration|"
        r"instructions?\s+(you\s+were|you\s+have\s+been|you've\s+been|given)|(own\s+)?(instructions|rules|guidelines))\b",
        r"\bwhat\s+(are|were|is|was)\s+your\s+(instructions|rules|system\s+prompt|prompt|configuration|guidelines)\b",
        r"\bsystem\s+prompt\b",
        r"\bignore\s+(your|all|any|the)?\s*(previous|prior|above|earlier)\s+instructions\b",
        r"\b(word\s+for\s+word|verbatim)\b.{0,40}\b(instructions|prompt|rules)\b|"
        r"\b(instructions|prompt|rules)\b.{0,40}\b(word\s+for\s+word|verbatim)\b",
    ]
]


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""
    canned_response: str = ""


def check_user_input(question: str) -> GuardResult:
    """Block empty, oversized and prompt-extraction inputs; pass everything else."""
    q = (question or "").strip()
    if not q:
        return GuardResult(True, "empty", "Please enter a question about the Cerulean Systems documents.")
    if len(q) > MAX_QUESTION_CHARS:
        return GuardResult(
            True, "too_long",
            f"Your question is too long ({len(q)} characters; the limit is {MAX_QUESTION_CHARS}). "
            "Please shorten it.",
        )
    for rx in _EXTRACTION_PATTERNS:
        if rx.search(q):
            log.warning("input guard blocked prompt-extraction attempt (pattern %r)", rx.pattern[:40])
            return GuardResult(True, "prompt_extraction", EXTRACTION_REFUSAL)
    return GuardResult(False)


# --------------------------------------------------------------------------- #
# 3. Output checks
# --------------------------------------------------------------------------- #
SAFE_INJECTION_MESSAGE = (
    "The retrieved document contains embedded text addressed to AI assistants, "
    "which was disregarded. Please re-ask your question; if this message persists, "
    "the document should be reviewed."
)
GENERIC_REFUSAL = (
    "I can't help with that request. I can explain the legitimate process described "
    "in the company documents if that would be useful."
)

_FIGURE_PATTERNS = [
    re.compile(r"SAR\s?[\d,]+(?:\.\d+)?", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"\b\d+\s+(?:calendar|working)\s+days?\b", re.I),
]
_FRAMING_WORDS = ("instruction", "disregard", "ignored", "ignore", "embedded", "injection",
                  "addressed to", "directive", "did not follow", "not followed")
_REFUSAL_LEAK_RE = re.compile(r"\bSAR\b|\d{3,}|\bthreshold|\bretrospective|\bemergency", re.I)
_SYSTEM_PROMPT_SHINGLE = 12


@dataclass
class VerificationResult:
    parsed: AnswerSchema
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False
    blocked_reason: str = ""


def normalise(text: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _context_digit_set(chunks: list[Chunk]) -> set[str]:
    out: set[str] = set()
    for c in chunks:
        for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", c.text):
            out.add(_digits(m.group(0)))
    return out


def _framed_as_disregarded(norm_answer: str, position: int) -> bool:
    window = norm_answer[max(0, position - 250): position + 250]
    return any(w in window for w in _FRAMING_WORDS)


def verify_answer(
    parsed: AnswerSchema,
    retrieved: list[Chunk],
    system_prompt: str | None = None,
) -> VerificationResult:
    """Apply the deterministic post-generation checks (a) to (f) from the design."""
    fixed = parsed.model_copy(deep=True)
    warnings: list[str] = []
    blocked = False
    blocked_reason = ""

    context_ids = {c.document_id for c in retrieved}
    norm_answer = normalise(fixed.answer)

    # (a) citations must point at documents that were actually in the context
    kept = []
    for cit in fixed.citations:
        if cit.document_id in context_ids:
            kept.append(cit)
        else:
            warnings.append(f"dropped citation to {cit.document_id}: not in retrieved context")
    fixed.citations = kept

    # (b) answer must not repeat an injection payload as fact
    for chunk in retrieved:
        if not chunk.has_injection:
            continue
        for payload in chunk.injection_payloads:
            norm_payload = normalise(payload)
            if len(norm_payload) < MIN_PAYLOAD_CHARS:
                continue
            pos = norm_answer.find(norm_payload)
            if pos >= 0 and not _framed_as_disregarded(norm_answer, pos):
                blocked = True
                blocked_reason = f"answer repeats injected text from {chunk.chunk_id}"
                warnings.append(blocked_reason)
                log.warning("%s: %r", blocked_reason, payload[:80])
                break
        if blocked:
            break
    if blocked:
        fixed.answer = SAFE_INJECTION_MESSAGE
        fixed.injection_noticed = True

    # (c) numeric grounding: figures in the answer should exist in the context
    if not blocked and fixed.decision != "refused":
        ctx_digits = _context_digit_set(retrieved)
        calc_text = " ".join(fixed.assumptions) + " " + " ".join(
            ln for ln in fixed.answer.splitlines() if "=" in ln
        )
        calc_digits = {_digits(m.group(0)) for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", calc_text)}
        seen: set[str] = set()
        for rx in _FIGURE_PATTERNS:
            for m in rx.finditer(fixed.answer):
                fig = m.group(0).strip()
                d = _digits(fig)
                if not d or d in seen:
                    continue
                seen.add(d)
                if d in ctx_digits or d in calc_digits:
                    continue
                warnings.append(f"figure '{fig}' not found in retrieved context")

    # (d) a positive answer with nothing to cite is not grounded
    if not blocked and fixed.decision == "answer" and not fixed.citations:
        fixed.decision = "insufficient_evidence"
        warnings.append("answer had no valid citations; downgraded to insufficient_evidence")

    # (e) refusals must not leak policy details
    if fixed.decision == "refused" and _REFUSAL_LEAK_RE.search(fixed.answer):
        warnings.append("refusal contained policy details; replaced with generic refusal")
        fixed.answer = GENERIC_REFUSAL

    # (f) the system prompt must never appear in an answer
    if system_prompt and not blocked:
        words = normalise(system_prompt).split()
        answer_norm = normalise(fixed.answer)
        for i in range(0, max(0, len(words) - _SYSTEM_PROMPT_SHINGLE + 1)):
            shingle = " ".join(words[i: i + _SYSTEM_PROMPT_SHINGLE])
            if shingle in answer_norm:
                blocked = True
                blocked_reason = "answer reproduced part of the system prompt"
                warnings.append(blocked_reason)
                fixed.decision = "refused"
                fixed.answer = EXTRACTION_REFUSAL
                log.warning(blocked_reason)
                break

    return VerificationResult(parsed=fixed, warnings=warnings, blocked=blocked, blocked_reason=blocked_reason)
