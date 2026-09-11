"""Run eval/questions.yaml through the pipeline and write a results table.

Usage:
  python scripts/run_eval.py --model qwen2.5:7b-instruct
  python scripts/run_eval.py --only Q4,Q5 --tag debug
  python scripts/run_eval.py --repeat 2          # stability: run every question twice

Writes eval/results/<timestamp>_<model>[_tag].json (full AnswerResults + scores)
and the matching .md (summary table, per-question answers, aggregate stats).

Scoring is generic (decision, cited docs, regex must/must-not, conflict flag,
injection flag, clarification count). Nothing here feeds back into the pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

from cerulean_rag.config import get_settings
from cerulean_rag.ingest import load_chunks_jsonl
from cerulean_rag.logging_setup import setup_logging
from cerulean_rag.models import AnswerResult
from cerulean_rag.pipeline import ask

CHECK_NAMES = ["decision", "docs", "contains", "not_contains", "conflict", "injection", "clarification"]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def searchable_text(result: AnswerResult) -> str:
    p = result.parsed
    parts = [p.answer]
    for c in p.conflicts:
        parts += [c.topic, *c.positions, c.resolution, c.reasoning]
    parts += p.clarification_options + p.assumptions
    return "\n".join(x for x in parts if x)


def score(spec: dict, result: AnswerResult, flagged_chunk_ids: set[str]) -> dict:
    p = result.parsed
    cited = {c.document_id for c in p.citations}
    retrieved_docs = {r["document_id"] for r in result.retrieved}
    text_all = searchable_text(result)
    text_answer = p.answer

    checks: dict[str, bool] = {}
    failed: list[str] = []

    checks["decision"] = p.decision in spec.get("expected_decision", [])
    if not checks["decision"]:
        failed.append(f"decision={p.decision} not in {spec.get('expected_decision')}")

    missing_docs = [d for d in spec.get("expected_docs", []) if d not in cited]
    checks["docs"] = not missing_docs
    if missing_docs:
        failed.append(f"not cited: {missing_docs}")

    miss = [rx for rx in spec.get("must_contain", []) if not re.search(rx, text_all, re.I)]
    checks["contains"] = not miss
    if miss:
        failed.append(f"missing: {miss}")

    hit = [rx for rx in spec.get("must_not_contain", []) if re.search(rx, text_answer, re.I)]
    checks["not_contains"] = not hit
    if hit:
        failed.append(f"forbidden present: {hit}")

    if spec.get("expected_conflict"):
        checks["conflict"] = bool(p.conflicts) or p.decision == "conflict_resolved"
        if not checks["conflict"]:
            failed.append("no conflict recorded")
    else:
        checks["conflict"] = True

    if spec.get("expected_injection_noticed"):
        flagged_retrieved = any(r["chunk_id"] in flagged_chunk_ids for r in result.retrieved)
        checks["injection"] = p.injection_noticed or not flagged_retrieved
        if not checks["injection"]:
            failed.append("injection present in context but not flagged by model")
    else:
        checks["injection"] = True

    n_min = spec.get("min_clarification_options")
    if n_min:
        checks["clarification"] = len(p.clarification_options) >= n_min
        if not checks["clarification"]:
            failed.append(f"only {len(p.clarification_options)} clarification options (< {n_min})")
    else:
        checks["clarification"] = True

    expected_docs = spec.get("expected_docs", [])
    retrieval_recall = 1.0 if not expected_docs else sum(d in retrieved_docs for d in expected_docs) / len(expected_docs)

    return {
        "pass": all(checks.values()),
        "checks": checks,
        "failed": failed,
        "cited_docs": sorted(cited),
        "retrieved_docs": sorted(retrieved_docs),
        "retrieval_recall": retrieval_recall,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_s(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}"


def write_markdown(path: Path, runs: list[dict], meta: dict) -> None:
    lines: list[str] = []
    lines.append(f"# Evaluation run — {meta['model']}")
    lines.append("")
    lines.append(f"- started: {meta['started']}  ·  finished: {meta['finished']}")
    lines.append(f"- model: `{meta['model']}`  ·  embed: `{meta['embed_model']}`  ·  as-of date: {meta['as_of_date']}")
    lines.append(f"- settings: TOP_K={meta['top_k']}, SIM_THRESHOLD={meta['sim_threshold']}, USE_BM25={meta['use_bm25']}, "
                 f"NUM_CTX={meta['num_ctx']}, TEMPERATURE={meta['temperature']}, SEED={meta['seed']}")
    lines.append(f"- questions: {meta['n_questions']}  ·  repeats: {meta['repeat']}  ·  prompt hash: `{meta['prompt_hash']}`")
    lines.append(f"- hardware: {meta['hardware']}")
    lines.append("")

    lines.append("## Summary table")
    lines.append("")
    lines.append("| id | run | category | decision | conf | cited docs | latency s | gen tok/s | result | failed checks |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        res = r["result"]
        sc = r["score"]
        tps = r.get("gen_tokens_per_s")
        lines.append(
            f"| {r['id']} | {r['run']} | {r['category']} | {res['parsed']['decision']} | {res['confidence']} | "
            f"{', '.join(sc['cited_docs']) or '—'} | {_fmt_s(res['timings_ms'].get('total_ms'))} | "
            f"{'-' if tps is None else f'{tps:.1f}'} | {'PASS' if sc['pass'] else 'FAIL'} | "
            f"{'; '.join(sc['failed']).replace('|', '/') or '—'} |"
        )
    lines.append("")

    # aggregates
    first_runs = [r for r in runs if r["run"] == 1]
    n = len(first_runs)
    n_pass = sum(r["score"]["pass"] for r in first_runs)
    latencies = [r["result"]["timings_ms"].get("total_ms", 0) / 1000 for r in first_runs]
    gen_lat = [r["result"]["timings_ms"].get("generation_ms", 0) / 1000 for r in first_runs
               if r["result"]["timings_ms"].get("generation_ms", 0) > 0]
    recalls = [r["score"]["retrieval_recall"] for r in first_runs if r["expected_docs"]]
    tps_vals = [r["gen_tokens_per_s"] for r in first_runs if r.get("gen_tokens_per_s")]
    pps_vals = [r["prompt_tokens_per_s"] for r in first_runs if r.get("prompt_tokens_per_s")]
    warn_counter: Counter[str] = Counter()
    for r in first_runs:
        for w in r["result"]["warnings"]:
            warn_counter[re.sub(r"'[^']*'|\d+", "…", w)[:70]] += 1
    by_cat: dict[str, list[bool]] = {}
    for r in first_runs:
        by_cat.setdefault(r["category"], []).append(r["score"]["pass"])
    safety_core = [r for r in first_runs if r["id"] in {"Q6", "Q7", "Q9", "Q10", "Q11"}]

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Pass rate (first run): {n_pass}/{n} = {100 * n_pass / max(n, 1):.0f}%**")
    if safety_core:
        lines.append(f"- Safety / hallucination core (Q6, Q7, Q9, Q10, Q11): "
                     f"{sum(r['score']['pass'] for r in safety_core)}/{len(safety_core)} pass")
    for cat, vals in sorted(by_cat.items()):
        lines.append(f"  - {cat}: {sum(vals)}/{len(vals)}")
    if latencies:
        lines.append(f"- Latency per question (total): mean {statistics.mean(latencies):.1f} s, "
                     f"median {statistics.median(latencies):.1f} s, max {max(latencies):.1f} s")
    if gen_lat:
        lines.append(f"- Generation only (questions that reached the LLM, n={len(gen_lat)}): "
                     f"mean {statistics.mean(gen_lat):.1f} s, median {statistics.median(gen_lat):.1f} s")
    if tps_vals:
        lines.append(f"- Model throughput: generation {statistics.mean(tps_vals):.2f} tok/s mean; "
                     f"prompt processing {statistics.mean(pps_vals):.1f} tok/s mean")
    if recalls:
        lines.append(f"- Retrieval recall of expected documents (in retrieved set, independent of citation): "
                     f"{100 * statistics.mean(recalls):.0f}% over {len(recalls)} questions with expectations")
    lines.append(f"- Warnings by type (first run):" + ("" if warn_counter else " none"))
    for w, c in warn_counter.most_common():
        lines.append(f"  - {c} × {w}")
    if meta["repeat"] > 1:
        lines.append("")
        lines.append("## Stability across repeats")
        lines.append("")
        by_id: dict[str, list[dict]] = {}
        for r in runs:
            by_id.setdefault(r["id"], []).append(r)
        stable_decision = stable_pass = 0
        for qid, rs in by_id.items():
            decisions = {x["result"]["parsed"]["decision"] for x in rs}
            passes = {x["score"]["pass"] for x in rs}
            stable_decision += len(decisions) == 1
            stable_pass += len(passes) == 1
            if len(decisions) > 1 or len(passes) > 1:
                lines.append(f"- {qid}: decisions {sorted(decisions)}, pass outcomes {sorted(passes)}")
        lines.append(f"- {stable_decision}/{len(by_id)} questions gave the same decision on every repeat; "
                     f"{stable_pass}/{len(by_id)} gave the same PASS/FAIL outcome.")
    lines.append("")

    lines.append("## Per-question detail")
    lines.append("")
    for r in runs:
        res = r["result"]
        p = res["parsed"]
        lines.append(f"### {r['id']} (run {r['run']}) — {r['category']} — {'PASS' if r['score']['pass'] else 'FAIL'}")
        lines.append("")
        lines.append(f"**Q:** {r['question']}")
        lines.append("")
        lines.append(f"**Expected:** decision ∈ {r['expected_decision']}; docs {r['expected_docs'] or '—'}. {r.get('notes', '')}")
        lines.append("")
        lines.append(f"**Decision:** `{p['decision']}` · confidence {res['confidence']} · "
                     f"injection_noticed={p['injection_noticed']} · method {res.get('generation_method')} · "
                     f"latency {_fmt_s(res['timings_ms'].get('total_ms'))} s")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        for ln in (p["answer"] or "(empty)").splitlines():
            lines.append(f"> {ln}")
        lines.append("")
        if p["citations"]:
            lines.append("**Citations:** " + "; ".join(
                f"{c['document_id']} §{c['section']}" + (f' — "{c["quote"][:90]}"' if c.get("quote") else "")
                for c in p["citations"]))
            lines.append("")
        if p["conflicts"]:
            lines.append("**Conflicts:**")
            for c in p["conflicts"]:
                lines.append(f"- {c['topic']}: {' / '.join(c['positions'])} → {c['resolution']}. {c['reasoning']}")
            lines.append("")
        if p["clarification_options"]:
            lines.append("**Clarification options:** " + " | ".join(p["clarification_options"]))
            lines.append("")
        if p["assumptions"]:
            lines.append("**Assumptions:** " + "; ".join(p["assumptions"]))
            lines.append("")
        lines.append("**Retrieved:** " + ", ".join(
            f"{x['document_id']} §{x['section'][:30]} ({'-' if x.get('vector_sim') is None else x['vector_sim']})"
            for x in res["retrieved"]) if res["retrieved"] else "**Retrieved:** (none — refused before retrieval)")
        lines.append("")
        if res["warnings"]:
            lines.append("**Warnings:** " + " | ".join(res["warnings"]))
            lines.append("")
        if r["score"]["failed"]:
            lines.append("**Failed checks:** " + "; ".join(r["score"]["failed"]))
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_questions(path: Path, only: set[str] | None) -> list[dict]:
    specs = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        raise SystemExit(f"{path}: expected a non-empty list of questions")
    if only:
        specs = [s for s in specs if s["id"] in only]
        missing = only - {s["id"] for s in specs}
        if missing:
            raise SystemExit(f"unknown question ids: {sorted(missing)}")
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation set through the pipeline.")
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--model", default=None, help="generation model tag (default: GEN_MODEL)")
    parser.add_argument("--out-dir", default="eval/results")
    parser.add_argument("--only", default=None, help="comma-separated ids, e.g. Q1,Q4")
    parser.add_argument("--tag", default=None, help="label appended to the output file names")
    parser.add_argument("--repeat", type=int, default=1, help="run every question this many times")
    parser.add_argument("--hardware", default="Intel Core i5-1135G7 (4C/8T), 19.8 GB RAM, no GPU, Windows 11",
                        help="free-text hardware description written into the report")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"GEN_MODEL": args.model})
    setup_logging(settings.LOG_LEVEL, settings.LOG_FILE)

    from cerulean_rag.prompts import prompt_hash, render_system_prompt

    only = {x.strip() for x in args.only.split(",")} if args.only else None
    specs = load_questions(Path(args.questions), only)
    flagged = {c.chunk_id for c in load_chunks_jsonl(Path(settings.CHUNKS_FILE)) if c.has_injection}

    started = datetime.now()
    runs: list[dict] = []
    total = len(specs) * args.repeat
    i = 0
    for rep in range(1, args.repeat + 1):
        for spec in specs:
            i += 1
            t0 = time.perf_counter()
            print(f"[{i}/{total}] {spec['id']} run {rep}: {spec['question'][:70]}", flush=True)
            result = ask(spec["question"], settings=settings)
            sc = score(spec, result, flagged)
            ts = result.token_stats or {}
            gen_tps = (ts["eval_count"] / (ts["eval_duration_ns"] / 1e9)) if ts.get("eval_duration_ns") else None
            prompt_tps = (ts["prompt_eval_count"] / (ts["prompt_eval_duration_ns"] / 1e9)) if ts.get("prompt_eval_duration_ns") else None
            runs.append({
                "id": spec["id"], "run": rep, "category": spec.get("category", ""),
                "question": spec["question"], "expected_decision": spec.get("expected_decision", []),
                "expected_docs": spec.get("expected_docs", []), "notes": spec.get("notes", ""),
                "result": result.model_dump(mode="json"), "score": sc,
                "gen_tokens_per_s": gen_tps, "prompt_tokens_per_s": prompt_tps,
            })
            print(f"      -> {result.parsed.decision:22} {'PASS' if sc['pass'] else 'FAIL'} "
                  f"({time.perf_counter() - t0:.0f}s) {'; '.join(sc['failed'])}", flush=True)

    finished = datetime.now()
    meta = {
        "model": settings.GEN_MODEL, "embed_model": settings.EMBED_MODEL,
        "as_of_date": settings.AS_OF_DATE.isoformat(), "top_k": settings.TOP_K,
        "sim_threshold": settings.SIM_THRESHOLD, "use_bm25": settings.USE_BM25,
        "num_ctx": settings.NUM_CTX, "temperature": settings.TEMPERATURE, "seed": settings.SEED,
        "n_questions": len(specs), "repeat": args.repeat,
        "prompt_hash": prompt_hash(render_system_prompt(settings.AS_OF_DATE)),
        "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
        "hardware": args.hardware,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{started.strftime('%Y%m%d_%H%M')}_{re.sub(r'[^A-Za-z0-9.]+', '-', settings.GEN_MODEL)}"
    if args.tag:
        stem += f"_{re.sub(r'[^A-Za-z0-9.-]+', '-', args.tag)}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps({"meta": meta, "runs": runs}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, runs, meta)

    first = [r for r in runs if r["run"] == 1]
    n_pass = sum(r["score"]["pass"] for r in first)
    print(f"\npass rate: {n_pass}/{len(first)}   wall time: {(finished - started).total_seconds() / 60:.1f} min")
    print(f"wrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
