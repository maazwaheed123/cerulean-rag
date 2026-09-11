"""The question-answering pipeline: ``ask(question) -> AnswerResult``.

Order of operations for one question (deterministic steps in plain Python,
exactly one LLM call in the middle):

0. input guard          empty / too long / prompt extraction -> refuse, no LLM
1-5. retrieval          hybrid search, fusion, signals, metadata notes
6. prompt assembly      fixed system prompt + per-question human message
7. generation           structured JSON answer (one retry path inside)
8. verification         citations, injection echo, figures, refusal leaks,
                        system-prompt leaks; confidence score
9. logging              one JSON line per question in logs/queries.jsonl
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from cerulean_rag.config import Settings, get_settings
from cerulean_rag.generation import generate
from cerulean_rag.models import AnswerResult, AnswerSchema, RetrievalBundle
from cerulean_rag.prompts import build_messages, prompt_hash, render_system_prompt
from cerulean_rag.retrieval import GOOD_SIM_MARGIN, Retriever, get_retriever
from cerulean_rag.security import check_user_input, verify_answer

log = logging.getLogger(__name__)


def compute_confidence(parsed: AnswerSchema, best_sim: float | None, warnings: list[str],
                       blocked: bool, threshold: float) -> str:
    """Three-level confidence from retrieval strength, citations and verification outcome."""
    if blocked or parsed.decision in {"insufficient_evidence", "refused", "needs_clarification"}:
        return "low"
    if any(("leak" in w) or ("repeats" in w) or ("failed" in w) for w in warnings):
        return "low"
    if best_sim is None or best_sim < threshold:
        return "low"
    distinct_docs = {c.document_id for c in parsed.citations}
    if best_sim >= threshold + GOOD_SIM_MARGIN and distinct_docs and not warnings:
        return "high"
    return "medium"


def _append_query_log(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:  # logging must never break answering
        log.warning("could not write query log %s: %s", path, exc)


def _log_record(result: AnswerResult, bundle: RetrievalBundle | None, s: Settings, system_prompt: str) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": result.question,
        "sub_queries": result.sub_queries,
        "retrieved": result.retrieved,
        "best_sim": None if bundle is None else bundle.best_sim,
        "signals": [] if bundle is None else bundle.signals,
        "decision": result.parsed.decision,
        "citations": [c.model_dump() for c in result.parsed.citations],
        "conflicts": [c.model_dump() for c in result.parsed.conflicts],
        "injection_noticed": result.parsed.injection_noticed,
        "confidence": result.confidence,
        "blocked": result.blocked,
        "warnings": result.warnings,
        "generation_method": result.generation_method,
        "timings_ms": result.timings_ms,
        "model": result.model,
        "prompt_hash": prompt_hash(system_prompt),
        "token_stats": result.token_stats,
        "answer": result.parsed.answer,
    }


def ask(question: str, settings: Settings | None = None, retriever: Retriever | None = None) -> AnswerResult:
    """Answer one question end to end. Never raises for model-output problems."""
    s = settings or get_settings()
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    system_prompt = render_system_prompt(s.AS_OF_DATE)

    # 0. input guard
    t0 = time.perf_counter()
    guard = check_user_input(question)
    timings["guard_ms"] = (time.perf_counter() - t0) * 1000
    if guard.blocked:
        parsed = AnswerSchema(decision="refused", answer=guard.canned_response)
        timings.update(retrieval_ms=0.0, generation_ms=0.0, verify_ms=0.0)
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        result = AnswerResult(
            question=question, sub_queries=[], parsed=parsed, confidence="low",
            warnings=[f"input guard: {guard.reason}"], retrieved=[], timings_ms=timings,
            model=s.GEN_MODEL, blocked=True, generation_method="none",
        )
        log.info("input guard refused question (%s), no LLM call", guard.reason)
        _append_query_log(Path(s.QUERY_LOG), _log_record(result, None, s, system_prompt))
        return result

    # 1-5. retrieval
    t0 = time.perf_counter()
    r = retriever or get_retriever()
    bundle = r.retrieve(question)
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # 6. prompt
    t0 = time.perf_counter()
    messages = build_messages(bundle, s.AS_OF_DATE)
    timings["prompt_ms"] = (time.perf_counter() - t0) * 1000
    prompt_chars = sum(len(m.content) for m in messages)

    # 7. generation
    t0 = time.perf_counter()
    gen = generate(messages, s)
    timings["generation_ms"] = (time.perf_counter() - t0) * 1000

    # 8. verification + confidence
    t0 = time.perf_counter()
    verification = verify_answer(gen.parsed, [rc.chunk for rc in bundle.chunks], system_prompt=system_prompt)
    warnings = gen.warnings + verification.warnings
    parsed = verification.parsed
    if any(rc.chunk.has_injection for rc in bundle.chunks) and not parsed.injection_noticed:
        warnings.append("retrieved context contained embedded instructions; model did not flag them")
    confidence = compute_confidence(parsed, bundle.best_sim, warnings, verification.blocked, s.SIM_THRESHOLD)
    timings["verify_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = (time.perf_counter() - t_start) * 1000

    result = AnswerResult(
        question=question,
        sub_queries=bundle.sub_queries,
        parsed=parsed,
        confidence=confidence,
        warnings=warnings,
        retrieved=[rc.summary() for rc in bundle.chunks],
        timings_ms=timings,
        model=s.GEN_MODEL,
        blocked=verification.blocked,
        generation_method=gen.method,
        signals=bundle.signals,
        prompt_chars=prompt_chars,
        token_stats=gen.token_stats,
    )
    log.info(
        "answered %r: decision=%s confidence=%s citations=%d warnings=%d gen=%.0fs total=%.0fs",
        question[:60], parsed.decision, confidence, len(parsed.citations), len(warnings),
        timings["generation_ms"] / 1000, timings["total_ms"] / 1000,
    )
    _append_query_log(Path(s.QUERY_LOG), _log_record(result, bundle, s, system_prompt))
    return result
