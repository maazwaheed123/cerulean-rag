"""Debug aid: show what retrieval hands to the prompt for a question, without the LLM.

Usage:
  python scripts/retrieval_check.py "What is the limit?"            # fused hybrid results
  python scripts/retrieval_check.py "annual leave" --vector-only     # dense side alone
  python scripts/retrieval_check.py "..." --k 12

Vector scores are cosine similarities (1 - Chroma cosine distance).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.console import Console
from rich.table import Table

from cerulean_rag.config import get_settings
from cerulean_rag.ingest import open_vector_store
from cerulean_rag.retrieval import Retriever


def _first_line(text: str, n: int = 70) -> str:
    body = text.split("\n", 1)[1] if "\n" in text else text
    return body.strip().replace("\n", " ")[:n]


def vector_only(question: str, k: int, console: Console) -> None:
    store = open_vector_store(get_settings())
    t0 = time.perf_counter()
    hits = store.similarity_search_with_score(question, k=k)
    table = Table(title=f"vector top-{k} for {question!r}  ({time.perf_counter() - t0:.2f}s)")
    for col in ("#", "sim", "document", "section", "first 70 chars"):
        table.add_column(col, justify="right" if col in ("#", "sim") else "left")
    for i, (doc, dist) in enumerate(hits, 1):
        md = doc.metadata
        table.add_row(str(i), f"{1 - dist:.3f}", md["document_id"], md["section_label"], _first_line(doc.page_content))
    console.print(table)


def fused(question: str, k: int, console: Console) -> None:
    retriever = Retriever(get_settings())
    bundle = retriever.retrieve(question, top_k=k)

    console.print(f"[bold]question:[/bold] {question}")
    console.print(f"[bold]sub-queries:[/bold] {bundle.sub_queries or '(none)'}")
    table = Table(title=f"fused top-{k}  ({bundle.timings_ms.get('retrieval_total_ms', 0):.0f} ms)")
    for col, just in (("#", "right"), ("rrf", "right"), ("vec sim", "right"), ("bm25", "right"),
                      ("src", "left"), ("document", "left"), ("section", "left"), ("first 70 chars", "left")):
        table.add_column(col, justify=just)
    for i, rc in enumerate(bundle.chunks, 1):
        table.add_row(
            str(i), f"{rc.rrf_score:.4f}",
            "-" if rc.vector_sim is None else f"{rc.vector_sim:.3f}",
            "-" if rc.bm25_score is None else f"{rc.bm25_score:.2f}",
            "+".join(s[0] for s in rc.sources), rc.document_id, rc.chunk.section_label,
            _first_line(rc.chunk.text),
        )
    console.print(table)
    console.print(f"[bold]best vector similarity:[/bold] {bundle.best_sim:.3f}" if bundle.best_sim is not None
                  else "[bold]best vector similarity:[/bold] none")
    console.print("[bold]signals:[/bold]")
    for s in bundle.signals:
        console.print(f"  - {s}")
    console.print("[bold]metadata notes:[/bold]")
    for n in bundle.metadata_notes:
        console.print(f"  - {n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=None, help="final top-k (default: TOP_K setting)")
    parser.add_argument("--vector-only", action="store_true", help="dense search only, no BM25/fusion")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    console = Console(width=170)
    k = args.k or get_settings().TOP_K
    if args.vector_only:
        vector_only(args.question, k, console)
    else:
        fused(args.question, k, console)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
