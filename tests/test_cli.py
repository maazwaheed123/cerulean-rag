"""Offline CLI tests: rendering, REPL command handling, error translation."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from cerulean_rag.cli import build_parser, friendly_error, render_result, run_chat
from cerulean_rag.config import Settings
from cerulean_rag.models import AnswerResult, AnswerSchema, Citation, Conflict


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, width=120, force_terminal=False, color_system=None), buf


def _result() -> AnswerResult:
    return AnswerResult(
        question="q",
        parsed=AnswerSchema(
            decision="conflict_resolved",
            answer="The current price is **SAR 5,200** per month.",
            citations=[Citation(document_id="SALES-PL-2026", section="1. Subscription plans", quote="SAR 5,200")],
            conflicts=[Conflict(topic="price", positions=["SALES-PL-2026: 5,200", "SUP-FAQ-001: 4,500"],
                                resolution="5,200", reasoning="newer document supersedes")],
        ),
        confidence="high",
        warnings=["figure 'SAR 1' not found in retrieved context"],
        retrieved=[{"chunk_id": "x", "document_id": "SALES-PL-2026", "section": "1. Subscription plans",
                    "rrf": 0.03, "vector_sim": 0.81, "bm25": 7.6, "sources": ["bm25", "vector"]}],
        timings_ms={"retrieval_ms": 2600.0, "generation_ms": 235600.0},
        model="qwen2.5:7b-instruct",
        signals=["Best passage similarity: 0.83 (good)."],
    )


def test_render_result_shows_all_blocks() -> None:
    console, buf = _console()
    render_result(_result(), console, show_context=True)
    out = buf.getvalue()
    assert "CONFLICT RESOLVED" in out
    assert "SAR 5,200" in out
    assert "the documents disagreed" in out                      # plain-English gloss on the badge
    assert "Where this comes from" in out and "SALES-PL-2026" in out
    assert "Why the documents disagreed" in out and "SUP-FAQ-001: 4,500" in out
    # the audit warning is shown in plain English, not in its internal wording
    assert "does not appear in any retrieved passage" in out
    assert "not found in retrieved context" not in out
    assert "retrieval 2.6 s" in out and "generation 235.6 s" in out
    assert "Passages retrieved" in out and "0.810" in out


def test_parser_subcommands() -> None:
    a = build_parser().parse_args(["--model", "qwen2.5:3b-instruct", "ask", "hi", "--json", "--show-context"])
    assert a.command == "ask" and a.question == "hi" and a.json and a.show_context and a.model == "qwen2.5:3b-instruct"
    c = build_parser().parse_args(["chat"])
    assert c.command == "chat" and c.json is False


def test_friendly_errors() -> None:
    s = Settings(_env_file=None)
    assert "ingest.py" in friendly_error(RuntimeError("Collection 'cerulean_docs' not found in data/chroma; run `python scripts/ingest.py` first"), s)
    assert "ingest.py" in friendly_error(FileNotFoundError("data/chunks.jsonl not found"), s)
    assert "Ollama" in friendly_error(ConnectionError("ConnectError: [WinError 10061] connection actively refused"), s)
    assert "ollama pull" in friendly_error(RuntimeError("model 'x' not found, try pulling it first"), s)
    assert "Something went wrong" in friendly_error(ValueError("weird"), s)


def test_chat_commands_without_llm(monkeypatch) -> None:
    calls: list[str] = []

    def fake_ask(question, settings=None, retriever=None):
        calls.append(question)
        return _result()

    import cerulean_rag.pipeline as pipeline

    monkeypatch.setattr(pipeline, "ask", fake_ask)
    lines = iter(["/help", "/context", "/model qwen2.5:3b-instruct", "/model", "What is the price?",
                  "/context", "/json", "/bogus", "/exit"])
    console, buf = _console()
    rc = run_chat(Settings(_env_file=None), console, input_fn=lambda prompt: next(lines))
    out = buf.getvalue()
    assert rc == 0
    assert calls == ["What is the price?"]
    assert "Commands:" in out
    assert "no question answered yet" in out
    assert "model set to qwen2.5:3b-instruct" in out and "current model: qwen2.5:3b-instruct" in out
    assert "CONFLICT RESOLVED" in out
    assert "Passages retrieved" in out
    assert "json output on" in out
    assert "unknown command /bogus" in out
    assert out.rstrip().endswith("bye")


def test_chat_eof_exits_cleanly() -> None:
    def raise_eof(prompt):
        raise EOFError

    console, buf = _console()
    assert run_chat(Settings(_env_file=None), console, input_fn=raise_eof) == 0
    assert "bye" in buf.getvalue()


def test_plain_warning_translates_known_audit_messages() -> None:
    """Verification warnings are written for the log; the terminal shows English."""
    from cerulean_rag.cli import plain_warning

    cases = {
        "dropped conflict entry 'price': all positions state the same value":
            "the sources actually agree",
        "dropped citation to HR-POL-002: not in retrieved context":
            "Dropped a citation to HR-POL-002",
        "model did not flag embedded instructions; disclosure added by the system":
            "the system added that note",
        "input guard: prompt_extraction":
            "Refused before searching the documents",
        "answer repeats injected text from PROC-PRO-002#3":
            "instructions planted in PROC-PRO-002#3",
    }
    for raw, expected in cases.items():
        assert expected in plain_warning(raw), raw

    # anything the table does not cover must survive verbatim rather than vanish
    assert plain_warning("some future warning") == "some future warning"

