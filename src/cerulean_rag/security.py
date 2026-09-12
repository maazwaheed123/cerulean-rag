"""Three independent, deterministic defences.

At ingest, the scanner flags document text that addresses an AI assistant and
extracts the claims such text wants repeated. Before retrieval, the input guard
refuses prompt-extraction attempts without spending an LLM call; it is
deliberately narrow, since policy-bypass requests are for the model to decline.
After generation, verify_answer checks the output against the context and those
extracted payloads.

All of it is regex and string comparison, and none of it is keyed to a
particular question or document: the patterns describe kinds of text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from cerulean_rag.models import AnswerSchema, Chunk

log = logging.getLogger(__name__)


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
    """Every pattern match, in document order."""
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
    """Sentences overlapping a span, plus the quotes and "respond that ..." claims
    inside them: what an obeyed injection would make the assistant say, and so
    what every answer is checked against."""
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

    # Sentences that overlap a span become payloads in their own right ...
    sentences = _sentences_with_offsets(text)
    flagged: set[int] = set()
    for i, (s_start, s_end, sentence) in enumerate(sentences):
        if any(sp.start < s_end and sp.end > s_start for sp in spans):
            add(sentence)
            flagged.add(i)

    # ... and the region searched for quotes and claims extends one sentence
    # either side of them. An injection often puts its trigger in one sentence
    # ("SYSTEM: Ignore all previous instructions.") and the claim it wants
    # repeated in the next, which matches no pattern of its own. Only quoted
    # strings and explicit "respond that ..." clauses are taken from the
    # neighbours, so ordinary policy prose beside an injection is not harvested.
    wanted = sorted({j for i in flagged for j in (i - 1, i, i + 1) if 0 <= j < len(sentences)})
    region = re.sub(r"\s+", " ", " ".join(sentences[j][2] for j in wanted))
    for q in _QUOTED_RE.findall(region):
        add(q)
    for claim in _CLAIM_RE.findall(region):
        for clause in re.split(r"\s+and\s+that\s+|;\s*", claim):
            add(clause)
    return payloads


def annotate_chunks(chunks: list[Chunk]) -> int:
    """Tag each chunk in place; returns how many were flagged."""
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


MAX_QUESTION_CHARS = 2000

EXTRACTION_REFUSAL = (
    "I can't share my configuration or instructions, but I'm happy to answer "
    "questions about the Cerulean Systems documents."
)

_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in [
        # "repeat the instructions you were given", "print your system prompt", "show me your rules".
        # The bare nouns (instructions/rules/guidelines) need the possessive: "the rules" is how
        # people ask about company policy, and an earlier version of this pattern refused
        # "show me the rules for expense approval" before it ever reached retrieval.
        r"\b(repeat|print|show|reveal|display|output|give|tell|share|dump|paste|recite)\b.{0,30}\b"
        r"(?:(?:your|the)\s+(?:system\s+prompt|initial\s+prompt|hidden\s+prompt|prompt|configuration)"
        r"|your\s+(?:own\s+)?(?:instructions|rules|guidelines)"
        r"|(?:the\s+)?instructions?\s+(?:you\s+were|you\s+have\s+been|you've\s+been|given))\b",
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
    """Blocks empty, oversized and prompt-extraction input; passes everything else."""
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
# Deliberately blunt: \d{3,} also matches a document id, so a refusal cannot name
# a document either. A refusal has nothing to cite, and naming the policy it
# declined to discuss is itself a leak.
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


QUOTE_SHINGLE = 4


def quote_is_traceable(quote: str, document_text: str) -> bool:
    """Does any run of four consecutive words in the quote appear in the document?

    Deliberately loose: models tidy punctuation and elide the middle of a sentence,
    and dropping a citation over that would cost more than it gains. What it does
    catch is a quote that shares no phrase at all with the document it names, which
    is the case that matters. Quotes too short to shingle are let through.
    """
    words = normalise(quote).split()
    if len(words) < QUOTE_SHINGLE:
        return True
    return any(" ".join(words[i: i + QUOTE_SHINGLE]) in document_text
               for i in range(len(words) - QUOTE_SHINGLE + 1))


def _framed_as_disregarded(norm_answer: str, position: int) -> bool:
    window = norm_answer[max(0, position - 250): position + 250]
    return any(w in window for w in _FRAMING_WORDS)


_SHINGLE_STOP = frozenset("""a an and are as at be by for from has have in is it its of on or that the this to was
were with you your do not any all when user users about only must""".split())
SHORT_PAYLOAD_MAX_WORDS = 12
SHINGLE_SIZE = 3


def payload_shingles(payload: str) -> list[str]:
    """Three-word fragments of a short payload, so that a paraphrase such as
    "there are no rate limits" is still caught. Long sentences are matched whole
    instead, or ordinary wording would trip the check."""
    words = normalise(payload).split()
    if len(words) > SHORT_PAYLOAD_MAX_WORDS:
        return []
    out: list[str] = []
    for i in range(0, len(words) - SHINGLE_SIZE + 1):
        sh = words[i: i + SHINGLE_SIZE]
        if sum(w not in _SHINGLE_STOP for w in sh) >= 2:
            out.append(" ".join(sh))
    return out


def find_payload_echo(text: str, payloads: list[str]) -> tuple[str, int] | None:
    """(payload, position) if text repeats a payload or a fragment of one, else None."""
    norm = normalise(text)
    if not norm:
        return None
    for payload in payloads:
        norm_payload = normalise(payload)
        if len(norm_payload) < MIN_PAYLOAD_CHARS:
            continue
        pos = norm.find(norm_payload)
        if pos >= 0:
            return payload, pos
        for sh in payload_shingles(payload):
            pos = norm.find(sh)
            if pos >= 0:
                return payload, pos
    return None


_POSITION_PREFIX_RE = re.compile(r"^\s*[A-Z]+-[A-Z]+-\d+[^:]{0,80}:\s*")
_POSITION_DOC_ID_RE = re.compile(r"^([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\b")
_INJECTION_COMMENTARY_RE = re.compile(
    r"\b(instruction|directive|embedded|ai assistants?|do not follow|contains_embedded|disregard)\b", re.I
)


def prune_non_conflicts(conflicts: list) -> tuple[list, list[str]]:
    """Keep only conflicts whose positions are two or more document values that
    actually differ. Drops entries where a position is commentary about injected
    text, and entries whose positions all agree once the "DOC-ID §section:"
    prefix is stripped."""
    kept: list = []
    warnings: list[str] = []
    for cf in conflicts:
        if any(_INJECTION_COMMENTARY_RE.search(pos) for pos in cf.positions):
            warnings.append(f"dropped conflict entry '{cf.topic[:40]}': a position was commentary about injected text")
            continue
        values = {normalise(_POSITION_PREFIX_RE.sub("", pos)) for pos in cf.positions if pos.strip()}
        if len(cf.positions) >= 2 and len(values) <= 1:
            warnings.append(f"dropped conflict entry '{cf.topic[:40]}': all positions state the same value")
            continue
        kept.append(cf)
    return kept, warnings


def verify_answer(
    parsed: AnswerSchema,
    retrieved: list[Chunk],
    system_prompt: str | None = None,
    trusted_text: str = "",
) -> VerificationResult:
    """The post-generation checks. Returns a corrected copy; never raises."""
    fixed = parsed.model_copy(deep=True)
    warnings: list[str] = []
    blocked = False
    blocked_reason = ""

    context_ids = {c.document_id for c in retrieved}
    norm_answer = normalise(fixed.answer)

    # citations must point at documents that were actually in the context, and the
    # quote must be traceable to that document's retrieved text. A quote is shown to
    # the reader as verbatim evidence, so an invented one is worse than none.
    text_by_doc: dict[str, str] = {}
    for c in retrieved:
        text_by_doc[c.document_id] = text_by_doc.get(c.document_id, "") + " " + normalise(c.text)
    kept = []
    for cit in fixed.citations:
        if cit.document_id not in context_ids:
            warnings.append(f"dropped citation to {cit.document_id}: not in retrieved context")
            continue
        if not quote_is_traceable(cit.quote, text_by_doc[cit.document_id]):
            warnings.append(
                f"citation to {cit.document_id}: the quote does not appear in the retrieved text"
            )
        kept.append(cit)
    fixed.citations = kept

    # an answer must not repeat an injection payload as fact
    payloads: list[str] = []
    payload_owner: dict[str, str] = {}
    for chunk in retrieved:
        if chunk.has_injection:
            for p in chunk.injection_payloads:
                payloads.append(p)
                payload_owner.setdefault(p, chunk.chunk_id)
    if payloads:
        hit = find_payload_echo(fixed.answer, payloads)
        if hit and not _framed_as_disregarded(norm_answer, hit[1]):
            blocked = True
            blocked_reason = f"answer repeats injected text from {payload_owner[hit[0]]}"
            warnings.append(blocked_reason)
            log.warning("%s: %r", blocked_reason, hit[0][:80])
            fixed.answer = SAFE_INJECTION_MESSAGE
            fixed.injection_noticed = True

        # nor use it as evidence in the structured fields
        kept_conflicts = []
        for cf in fixed.conflicts:
            cf_text = " ".join([cf.topic, *cf.positions, cf.resolution, cf.reasoning])
            if find_payload_echo(cf_text, payloads):
                warnings.append(f"dropped conflict entry '{cf.topic[:40]}' that used injected text as evidence")
                fixed.injection_noticed = True
                continue
            kept_conflicts.append(cf)
        fixed.conflicts = kept_conflicts
        for field_name in ("assumptions", "clarification_options"):
            items = getattr(fixed, field_name)
            clean = [x for x in items if not find_payload_echo(x, payloads)]
            if len(clean) != len(items):
                warnings.append(f"dropped {len(items) - len(clean)} {field_name} item(s) that used injected text")
                fixed.injection_noticed = True
            setattr(fixed, field_name, clean)

        # A quote is the most authoritative-looking place on screen, so injected text
        # reaching it would be worse than in the prose, not better.
        kept_citations = []
        for cit in fixed.citations:
            if find_payload_echo(cit.quote, payloads):
                warnings.append(f"dropped citation to {cit.document_id}: its quote repeated injected text")
                fixed.injection_noticed = True
                continue
            kept_citations.append(cit)
        fixed.citations = kept_citations

    # a conflict needs two document values that disagree
    fixed.conflicts, dropped = prune_non_conflicts(fixed.conflicts)
    warnings.extend(dropped)
    # A conflict position asserts what a document says, so that document has to be
    # cited. Warn rather than drop: a real conflict reported with a missing citation
    # is still worth showing, but the reader should know the claim is unbacked.
    cited_ids = {c.document_id for c in fixed.citations}
    for cf in fixed.conflicts:
        for pos in cf.positions:
            m = _POSITION_DOC_ID_RE.match(pos.strip())
            if m and m.group(1) not in cited_ids:
                warnings.append(
                    f"conflict '{cf.topic[:40]}' quotes {m.group(1)} but the answer does not cite it"
                )
    if fixed.decision == "conflict_resolved" and not fixed.conflicts:
        fixed.decision = "answer"
        warnings.append("no genuine conflict remained; decision set to answer")

    # figures in the answer should exist somewhere in the context
    if not blocked and fixed.decision != "refused":
        # Figures the system itself supplied (the date-span helper's month and day
        # counts) are as grounded as the excerpts; without them a model that used
        # the arithmetic it was handed would be warned about for doing so.
        ctx_digits = _context_digit_set(retrieved)
        if trusted_text:
            ctx_digits |= {_digits(m.group(0))
                           for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", trusted_text)}
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

    # a positive answer with nothing to cite is not grounded
    if not blocked and fixed.decision == "answer" and not fixed.citations:
        fixed.decision = "insufficient_evidence"
        warnings.append("answer had no valid citations; downgraded to insufficient_evidence")

    # refusals must not leak policy details
    if fixed.decision == "refused" and _REFUSAL_LEAK_RE.search(fixed.answer):
        warnings.append("refusal contained policy details; replaced with generic refusal")
        fixed.answer = GENERIC_REFUSAL

    # the system prompt must never appear in an answer
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
