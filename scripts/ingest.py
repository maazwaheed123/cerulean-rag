"""Build the knowledge base from ./corpus into ./data (Chroma + chunks.jsonl).

Usage:  python scripts/ingest.py [--no-embed] [--json]
"""

import sys

from cerulean_rag.ingest import main

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
