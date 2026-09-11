"""Debug aid: print the top-k chunks for a question from VECTOR search only.

Usage:  python scripts/retrieval_check.py "annual leave entitlement" [--k 8]

Scores are cosine similarities (1 - Chroma cosine distance) between the
"search_query:"-prefixed question and the "search_document:"-prefixed chunk.
BM25 and fusion are added in Step 6; this script isolates the dense side.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    store = open_vector_store(settings)

    t0 = time.perf_counter()
    hits = store.similarity_search_with_score(args.question, k=args.k)
    elapsed = time.perf_counter() - t0

    table = Table(title=f"vector top-{args.k} for: {args.question!r}  ({elapsed:.2f}s)")
    table.add_column("#", justify="right")
    table.add_column("sim", justify="right")
    table.add_column("document")
    table.add_column("section")
    table.add_column("first 80 chars")
    best = None
    for i, (doc, distance) in enumerate(hits, 1):
        sim = 1.0 - distance
        best = sim if best is None else max(best, sim)
        md = doc.metadata
        first = doc.page_content.split("\n", 1)[1][:80] if "\n" in doc.page_content else doc.page_content[:80]
        table.add_row(str(i), f"{sim:.3f}", md.get("document_id", "?"), md.get("section_label", "?"), first)
    Console(width=150).print(table)
    print(f"best similarity: {best:.3f}" if best is not None else "no results")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
