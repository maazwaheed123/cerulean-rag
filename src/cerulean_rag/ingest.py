"""load -> chunk -> scan -> embed -> persist.

Writes data/chroma (one vector per chunk, cosine space) and data/chunks.jsonl
(the full Chunk records, which feed the BM25 index and act as the audit trail
for what was embedded). Always a full rebuild: 13 small files do not justify
incremental indexing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
from langchain_chroma import Chroma

from cerulean_rag.chunking import chunk_documents
from cerulean_rag.config import Settings, get_settings
from cerulean_rag.embeddings import get_embeddings
from cerulean_rag.loaders import load_corpus
from cerulean_rag.logging_setup import setup_logging
from cerulean_rag.models import Chunk
from cerulean_rag.security import annotate_chunks

log = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 32


@dataclass
class IngestStats:
    documents: int
    chunks: int
    flagged_chunks: int
    embed_seconds: float
    total_seconds: float
    chroma_dir: str
    chunks_file: str
    collection: str

    def summary(self) -> str:
        return (
            f"{self.documents} documents, {self.chunks} chunks, "
            f"{self.flagged_chunks} flagged for embedded instructions, "
            f"embed {self.embed_seconds:.1f} s, total {self.total_seconds:.1f} s"
        )


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n")
    log.info("wrote %d chunks to %s", len(chunks), path)


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python scripts/ingest.py` first")
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks


def open_vector_store(settings: Settings, create: bool = False) -> Chroma:
    """With create=False the collection must already exist, so a query path can
    never silently build an empty index."""
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    if not create:
        names = [c.name for c in client.list_collections()]
        if settings.COLLECTION_NAME not in names:
            raise RuntimeError(
                f"Collection {settings.COLLECTION_NAME!r} not found in {settings.CHROMA_DIR}; "
                "run `python scripts/ingest.py` first"
            )
    return Chroma(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding_function=get_embeddings(settings),
        collection_metadata={"hnsw:space": "cosine"},
        create_collection_if_not_exists=create,
    )


def _drop_collection(settings: Settings) -> None:
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    names = [c.name for c in client.list_collections()]
    if settings.COLLECTION_NAME in names:
        client.delete_collection(settings.COLLECTION_NAME)
        log.info("dropped existing collection %r", settings.COLLECTION_NAME)


def build_index(settings: Settings | None = None, embed: bool = True) -> IngestStats:
    """embed=False writes chunks.jsonl only, so it runs without Ollama."""
    s = settings or get_settings()
    t0 = time.perf_counter()

    docs = load_corpus(s.CORPUS_DIR)
    chunks = chunk_documents(docs)
    flagged = annotate_chunks(chunks)
    write_chunks_jsonl(chunks, Path(s.CHUNKS_FILE))

    embed_seconds = 0.0
    if embed:
        s.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _drop_collection(s)
        store = open_vector_store(s, create=True)
        t_embed = time.perf_counter()
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start: start + EMBED_BATCH_SIZE]
            store.add_texts(
                texts=[c.text_for_embedding for c in batch],
                metadatas=[c.to_metadata() for c in batch],
                ids=[c.chunk_id for c in batch],
            )
            log.info("embedded %d/%d chunks", min(start + EMBED_BATCH_SIZE, len(chunks)), len(chunks))
        embed_seconds = time.perf_counter() - t_embed
        count = store._collection.count()  # noqa: SLF001 - simplest reliable count
        if count != len(chunks):
            raise RuntimeError(f"Chroma holds {count} vectors but {len(chunks)} chunks were produced")

    stats = IngestStats(
        documents=len(docs),
        chunks=len(chunks),
        flagged_chunks=flagged,
        embed_seconds=embed_seconds,
        total_seconds=time.perf_counter() - t0,
        chroma_dir=str(s.CHROMA_DIR),
        chunks_file=str(s.CHUNKS_FILE),
        collection=s.COLLECTION_NAME,
    )
    log.info("ingest complete: %s", stats.summary())
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Cerulean RAG knowledge base.")
    parser.add_argument("--no-embed", action="store_true",
                        help="only write data/chunks.jsonl; skip Ollama embedding and Chroma")
    parser.add_argument("--rebuild", action="store_true",
                        help="accepted for clarity; the index is always rebuilt from scratch")
    parser.add_argument("--json", action="store_true", help="print stats as JSON")
    args = parser.parse_args(argv)

    s = get_settings()
    setup_logging(s.LOG_LEVEL, s.LOG_FILE)
    try:
        stats = build_index(s, embed=not args.no_embed)
    except Exception as exc:
        log.error("ingest failed: %s", exc)
        return 1
    print(json.dumps(asdict(stats), indent=2) if args.json else stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
