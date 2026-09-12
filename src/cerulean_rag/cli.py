"""Command-line interface: ``ask`` (one question) and ``chat`` (REPL).

    python -m cerulean_rag.cli ask "How much notice during probation?" [--json] [--show-context] [--model TAG]
    python -m cerulean_rag.cli chat

Each chat turn is independent: there is no conversation memory. That is a
deliberate choice (memory would let an earlier injected instruction persist
across turns and makes evaluation non-reproducible); it is listed in the README.

All failures are turned into one-line messages for the user; the traceback
goes to the log file only.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from cerulean_rag.config import Settings, get_settings
from cerulean_rag.logging_setup import setup_logging
from cerulean_rag.models import AnswerResult

log = logging.getLogger(__name__)

DECISION_STYLE: dict[str, tuple[str, str]] = {
    "answer": ("ANSWER", "bold green"),
    "conflict_resolved": ("CONFLICT RESOLVED", "bold yellow"),
    "needs_clarification": ("NEEDS CLARIFICATION", "bold cyan"),
    "insufficient_evidence": ("NOT IN KNOWLEDGE BASE", "bold magenta"),
    "refused": ("DECLINED", "bold red"),
}
CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}

HELP_TEXT = """Commands:
  /exit, /quit      leave the chat
  /context          show the passages retrieved for the last question
  /model <tag>      switch generation model for following questions (e.g. qwen2.5:3b-instruct)
  /json             toggle raw JSON output
  /help             this list
Each question is answered independently; there is no conversation memory."""


# --------------------------------------------------------------------------- #
# Preflight + error translation
# --------------------------------------------------------------------------- #
def ollama_problem(base_url: str) -> str | None:
    """Return a user-facing message if the Ollama server is not reachable, else None."""
    try:
        with urllib.request.urlopen(base_url, timeout=3) as resp:
            if resp.status == 200:
                return None
            return f"Ollama at {base_url} answered HTTP {resp.status}."
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return (
            f"Cannot reach Ollama at {base_url} ({exc.__class__.__name__}). "
            "Start it with `ollama serve` (or open the Ollama app), then retry. "
            "Set OLLAMA_BASE_URL in .env if it runs elsewhere."
        )


def friendly_error(exc: BaseException, settings: Settings) -> str:
    """Map an exception to one sentence the user can act on."""
    text = f"{exc.__class__.__name__}: {exc}"
    lower = text.lower()
    if "collection" in lower and "not found" in lower or "chunks.jsonl" in lower or "run `python scripts/ingest.py`" in lower:
        return "The knowledge base has not been built yet. Run `python scripts/ingest.py` first."
    if any(k in lower for k in ("connecterror", "connection refused", "actively refused", "11434", "connecttimeout", "remoteprotocolerror")):
        return ollama_problem(settings.OLLAMA_BASE_URL) or f"Ollama request failed: {exc}"
    if "model" in lower and ("not found" in lower or "pull" in lower):
        return f"Model {settings.GEN_MODEL!r} is not available in Ollama. Run `ollama pull {settings.GEN_MODEL}`."
    return f"Something went wrong ({exc.__class__.__name__}). Details are in {settings.LOG_FILE}."


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
# Each decision gets a plain-English gloss. The bare label ("INSUFFICIENT
# EVIDENCE") tells a first-time reader nothing about what the assistant did.
DECISION_GLOSS: dict[str, str] = {
    "answer": "answered from the retrieved documents",
    "conflict_resolved": "the documents disagreed; the conflict is set out below",
    "needs_clarification": "the question could mean several things; choose one below",
    "insufficient_evidence": "the documents do not contain this",
    "refused": "this request was declined",
}

# Verification warnings are written for the audit log, in the vocabulary of the
# code that raised them. Translate them for the terminal; anything unrecognised
# falls through unchanged rather than being hidden.
_WARNING_PLAIN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^dropped conflict entry '(?P<t>.*?)': a position was commentary about injected text$"),
     "Dropped a reported conflict about “{t}” - one side of it was commentary on planted text."),
    (re.compile(r"^dropped conflict entry '(?P<t>.*?)': all positions state the same value$"),
     "Dropped a reported conflict about “{t}” - the sources actually agree."),
    (re.compile(r"^dropped conflict entry '(?P<t>.*?)' that used injected text as evidence$"),
     "Dropped a reported conflict about “{t}” - it used planted text as evidence."),
    (re.compile(r"^dropped citation to (?P<d>\S+): not in retrieved context$"),
     "Dropped a citation to {d} - that document was not among the passages retrieved."),
    (re.compile(r"^dropped (?P<n>\d+) (?P<f>.+?) item\(s\) that used injected text$"),
     "Dropped {n} {f} item(s) that relied on text planted in a document."),
    (re.compile(r"^no genuine conflict remained; decision set to answer$"),
     "No real disagreement remained after checking, so this is reported as a plain answer."),
    (re.compile(r"^figure '(?P<f>.*?)' not found in retrieved context$"),
     "The figure {f} does not appear in any retrieved passage - treat it with caution."),
    (re.compile(r"^answer had no valid citations; downgraded to insufficient_evidence$"),
     "The answer cited nothing that could be checked, so it was downgraded to “not in knowledge base”."),
    (re.compile(r"^refusal contained policy details; replaced with generic refusal$"),
     "The refusal was leaking policy detail, so it was replaced with a plain one."),
    (re.compile(r"^answer repeats injected text from (?P<c>\S+)$"),
     "Blocked: the answer repeated instructions planted in {c}. A safe message was sent instead."),
    (re.compile(r"^answer reproduced part of the system prompt$"),
     "Blocked: the answer repeated part of the assistant's own instructions. A refusal was sent instead."),
    (re.compile(r"^model did not flag embedded instructions; disclosure added by the system$"),
     "The model did not mention the planted instructions it was shown, so the system added that note."),
    (re.compile(r"^structured output failed: .*$"),
     "The model's first reply did not match the required format, so it was asked again."),
    (re.compile(r"^input guard: (?P<k>.+)$"),
     "Refused before searching the documents (input guard: {k})."),
]


def plain_warning(w: str) -> str:
    """One audit warning rewritten for a human reader; unknown ones pass through."""
    for rx, template in _WARNING_PLAIN:
        m = rx.match(w.strip())
        if m:
            return template.format(**m.groupdict())
    return w


def _doc_lookup() -> dict:
    """document_id -> DocumentMeta, from the loaded retriever when available."""
    try:
        from cerulean_rag.retrieval import get_retriever

        return get_retriever().all_meta
    except Exception:  # rendering must work even when the index is missing
        return {}


def _heading(console: Console, text: str) -> None:
    console.print()
    console.print(f"[bold]{text}[/bold]")


def render_result(result: AnswerResult, console: Console, show_context: bool = False) -> None:
    p = result.parsed
    label, style = DECISION_STYLE.get(p.decision, (p.decision.upper(), "bold"))
    colour = style.split()[-1]
    conf_style = CONFIDENCE_STYLE.get(result.confidence, "white")
    gloss = DECISION_GLOSS.get(p.decision, "")

    # The verdict, what it means and how sure the system is all sit on the frame
    # of the answer itself, instead of on loose lines above it.
    console.print(
        Panel(
            Markdown(p.answer or "(empty answer)"),
            title=f"[{style}]{label}[/]  [dim]/[/dim]  confidence [{conf_style}]{result.confidence}[/]",
            title_align="left",
            subtitle=f"[dim]{gloss}[/dim]" if gloss else None,
            subtitle_align="left",
            border_style=colour,
            padding=(1, 2),
        )
    )

    if p.clarification_options:
        _heading(console, "Which did you mean?")
        for i, opt in enumerate(p.clarification_options, 1):
            console.print(f"  [cyan]{i}.[/cyan] {opt}")

    if p.conflicts:
        _heading(console, "Why the documents disagreed")
        for cf in p.conflicts:
            console.print(f"  [bold yellow]{cf.topic}[/bold yellow]")
            for pos in cf.positions:
                console.print(f"      [yellow]-[/yellow] {pos}")
            if cf.resolution:
                console.print(f"      [bold green]=>[/bold green] {cf.resolution}")
            if cf.reasoning:
                console.print(f"         [dim]{cf.reasoning}[/dim]")

    if p.citations:
        meta = _doc_lookup()
        _heading(console, "Where this comes from")
        seen: set[tuple[str, str]] = set()
        n = 0
        for c in p.citations:
            key = (c.document_id, c.section)
            if key in seen:
                continue
            seen.add(key)
            n += 1
            m = meta.get(c.document_id)
            console.print(f"  [cyan]{n}.[/cyan] [bold]{c.document_id}[/bold]" + (f"  {m.title}" if m else ""))
            detail = f"section {c.section}" if c.section else ""
            if m:
                detail += f"{'  /  ' if detail else ''}version {m.version}, effective {m.effective_date.isoformat()}"
            if detail:
                console.print(f"     [dim]{detail}[/dim]")
            if c.quote.strip():
                console.print(f'     [italic]"{c.quote.strip()}"[/italic]')

    if p.assumptions:
        _heading(console, "Assumptions made")
        for a in p.assumptions:
            console.print(f"  - {a}")

    if p.injection_noticed:
        _heading(console, "Security")
        console.print("  [yellow]A retrieved document contained instructions aimed at AI assistants.[/yellow]")
        console.print("  [dim]They were treated as ordinary text and not obeyed.[/dim]")

    if result.warnings:
        _heading(console, "What the system changed or flagged")
        for w in result.warnings:
            console.print(f"  [dim]- {plain_warning(w)}[/dim]")

    t = result.timings_ms
    console.print()
    console.print(
        f"[dim]retrieval {t.get('retrieval_ms', 0) / 1000:.1f} s / generation "
        f"{t.get('generation_ms', 0) / 1000:.1f} s / model {result.model}"
        f" / {len(result.retrieved)} passage{'' if len(result.retrieved) == 1 else 's'} used[/dim]"
    )
    if show_context:
        render_context(result, console)


def render_context(result: AnswerResult, console: Console) -> None:
    if not result.retrieved:
        console.print("[dim](no passages were retrieved for this question)[/dim]")
        return
    _heading(console, "Passages retrieved (best first)")
    table = Table(box=None, pad_edge=False, show_edge=False, header_style="dim")
    for col, just in (("#", "right"), ("document", "left"), ("section", "left"),
                      ("meaning", "right"), ("keyword", "right"), ("found by", "left")):
        table.add_column(col, justify=just)
    for i, rc in enumerate(result.retrieved, 1):
        table.add_row(
            str(i), rc["document_id"], rc["section"],
            "-" if rc.get("vector_sim") is None else f"{rc['vector_sim']:.3f}",
            "-" if rc.get("bm25") is None else f"{rc['bm25']:.1f}",
            " + ".join(rc.get("sources", [])),
        )
    console.print(table)
    console.print("[dim]  meaning = vector similarity (0-1)   keyword = BM25 score   "
                  "dash = that search missed it[/dim]")
    if result.signals:
        _heading(console, "What the system told the model before it answered")
        for s in result.signals:
            first, *rest = s.splitlines()
            console.print(f"  [dim]- {first}[/dim]")
            for line in rest:
                console.print(f"      [dim]{line.strip()}[/dim]")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def run_ask(question: str, settings: Settings, console: Console, as_json: bool, show_context: bool) -> int:
    from cerulean_rag.pipeline import ask

    try:
        result = ask(question, settings=settings)
    except Exception as exc:
        log.error("ask failed: %s", exc, exc_info=True)   # traceback -> log file only
        console.print(f"[red]{friendly_error(exc, settings)}[/red]")
        return 1
    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        render_result(result, console, show_context=show_context)
    return 0


def run_chat(settings: Settings, console: Console, as_json: bool = False,
             input_fn: Callable[[str], str] | None = None) -> int:
    """REPL. ``input_fn`` is injectable for tests; defaults to console.input."""
    from cerulean_rag.pipeline import ask

    read = input_fn or (lambda prompt: console.input(prompt))
    console.print("[bold]Cerulean Systems document assistant[/bold] — type a question, or /help for commands.")
    last: AnswerResult | None = None
    while True:
        try:
            line = read("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            cmd = cmd.lower()
            if cmd in ("/exit", "/quit"):
                console.print("bye")
                return 0
            if cmd == "/help":
                console.print(HELP_TEXT)
            elif cmd == "/json":
                as_json = not as_json
                console.print(f"json output {'on' if as_json else 'off'}")
            elif cmd == "/model":
                if not arg.strip():
                    console.print(f"current model: {settings.GEN_MODEL}")
                else:
                    settings = settings.model_copy(update={"GEN_MODEL": arg.strip()})
                    console.print(f"model set to {settings.GEN_MODEL} for following questions")
            elif cmd == "/context":
                if last is None:
                    console.print("no question answered yet")
                else:
                    render_context(last, console)
            else:
                console.print(f"unknown command {cmd}; try /help")
            continue
        try:
            last = ask(line, settings=settings)
        except Exception as exc:
            log.error("chat turn failed: %s", exc, exc_info=True)   # traceback -> log file only
            console.print(f"[red]{friendly_error(exc, settings)}[/red]")
            continue
        if as_json:
            print(last.model_dump_json(indent=2))
        else:
            render_result(last, console)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cerulean-rag", description="Ask questions about the Cerulean Systems documents.")
    parser.add_argument("--model", help="generation model tag (default: GEN_MODEL from .env)")
    parser.add_argument("--log-level", help="override LOG_LEVEL for this run")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ask = sub.add_parser("ask", help="answer one question and exit")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="print the full AnswerResult as JSON")
    p_ask.add_argument("--show-context", action="store_true", help="also list the retrieved passages")
    p_chat = sub.add_parser("chat", help="interactive question loop")
    p_chat.add_argument("--json", action="store_true", help="print JSON instead of formatted output")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"GEN_MODEL": args.model})
    setup_logging(args.log_level or settings.LOG_LEVEL, settings.LOG_FILE)
    # Console shows only errors; INFO/WARNING detail lives in the log file and
    # the CLI prints its own Warnings line from the AnswerResult.
    for h in logging.getLogger().handlers:
        if h.__class__.__name__ == "RichHandler":
            h.setLevel(logging.ERROR)
    console = Console()

    problem = ollama_problem(settings.OLLAMA_BASE_URL)
    if problem:
        console.print(f"[red]{problem}[/red]")
        return 2

    if args.command == "ask":
        return run_ask(args.question, settings, console, args.json, args.show_context)
    return run_chat(settings, console, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
