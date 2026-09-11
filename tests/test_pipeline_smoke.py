"""End-to-end smoke tests that need a running Ollama server.

Marked ``ollama``; conftest skips them automatically when the server is not
reachable. They also skip when the index has not been built, so a plain
`pytest` on a fresh clone never fails for environmental reasons.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cerulean_rag.config import get_settings

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def built_index(settings):
    if not Path(settings.CHROMA_DIR).exists() or not Path(settings.CHUNKS_FILE).exists():
        pytest.skip("index not built; run `python scripts/ingest.py`")
    return settings


def test_embedding_dimension(settings) -> None:
    from cerulean_rag.embeddings import get_embeddings

    vec = get_embeddings(settings).embed_query("annual leave")
    assert len(vec) == 768


def test_vector_store_has_all_chunks(built_index) -> None:
    from cerulean_rag.ingest import load_chunks_jsonl, open_vector_store

    chunks = load_chunks_jsonl(Path(built_index.CHUNKS_FILE))
    store = open_vector_store(built_index)
    assert store._collection.count() == len(chunks)  # noqa: SLF001


def test_direct_question_hits_expected_section(built_index) -> None:
    from cerulean_rag.ingest import open_vector_store

    store = open_vector_store(built_index)
    hits = store.similarity_search_with_score("annual leave entitlement per year", k=3)
    top_ids = [(d.metadata["document_id"], d.metadata["section_number"]) for d, _ in hits]
    assert ("HR-POL-002", "4.2") in top_ids
