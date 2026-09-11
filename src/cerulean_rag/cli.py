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
def _doc_lookup() -> dict:
    """document_id -> DocumentMeta, from the loaded retriever when available."""
    try:
        from cerulean_rag.retrieval import get_retriever

        return get_retriever().all_meta
    except Exception:  # rendering must work even when the index is missing
        return {}


def render_result(result: AnswerResult, console: Console, show_context: bool = False) -> None:
    p = result.parsed
    label, style = DECISION_STYLE.get(p.decision, (p.decision.upper(), "bold"))
    conf_style = CONFIDENCE_STYLE.get(result.confidence, "white")
    console.print(f"[{style}]{label}[/]   confidence: [{conf_style}]{result.confidence}[/]")
    console.print(Panel(Markdown(p.answer or "(empty answer)"), border_style=style.split()[-1]))

    if p.citations:
        meta = _doc_lookup()
        console.print("[bold]Sources:[/bold]")
        seen: set[tuple[str, str]] = set()
        for c in p.citations:
            key = (c.document_id, c.section)
            if key in seen:
                continue
            seen.add(key)
            m = meta.get(c.document_id)
            head = (f"{c.document_id} v{m.version} — {m.title} (effective {m.effective_date.isoformat()})"
                    if m else c.document_id)
            section = f" §{c.section}" if c.section else ""
            quote = f'  "{c.quote.strip()}"' if c.quote.strip() else ""
            console.print(f"  • {head}{section}{quote}")

    if p.conflicts:
        console.print("[bold yellow]Conflicts found:[/bold yellow]")
        for cf in p.conflicts:
            console.print(f"  • [bold]{cf.topic}[/bold]")
            for pos in cf.positions:
                console.print(f"      - {pos}")
            if cf.resolution:
                console.print(f"      resolution: {cf.resolution}")
            if cf.reasoning:
                console.print(f"      reasoning: {cf.reasoning}")

    if p.clarification_options:
        console.print("[bold cyan]Possible interpretations:[/bold cyan]")
        for i, opt in enumerate(p.clarification_options, 1):
            console.print(f"  {i}. {opt}")

    if p.assumptions:
        console.print("[bold]Assumptions:[/bold] " + "; ".join(p.assumptions))
    if p.injection_noticed:
        console.print("[yellow]Note: a retrieved document contained embedded instructions addressed to AI assistants; they were treated as text.[/yellow]")
    if result.warnings:
        console.print("[dim]Warnings: " + " | ".join(result.warnings) + "[/dim]")

    t = result.timings_ms
    console.print(
        f"[dim]retrieval {t.get('retrieval_ms', 0) / 1000:.1f} s · generation {t.get('generation_ms', 0) / 1000:.1f} s"
        f" · model {result.model}[/dim]"
    )
    if show_context:
        render_context(result, console)


def render_context(result: AnswerResult, console: Console) -> None:
    if not result.retrieved:
        console.print("[dim](no passages were retrieved for this question)[/dim]")
        return
    table = Table(title="Retrieved passages (fused ranking)")
    for col, just in (("#", "right"), ("document", "left"), ("section", "left"),
                      ("vec sim", "right"), ("bm25", "right"), ("via", "left")):
        table.add_column(col, justify=just)
    for i, rc in enumerate(result.retrieved, 1):
        table.add_row(
            str(i), rc["document_id"], rc["section"],
            "-" if rc.get("vector_sim") is None else f"{rc['vector_sim']:.3f}",
            "-" if rc.get("bm25") is None else f"{rc['bm25']:.1f}",
            "+".join(rc.get("sources", [])),
        )
    console.print(table)
    if result.signals:
        console.print("[dim]signals: " + " ".join(result.signals) + "[/dim]")


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
