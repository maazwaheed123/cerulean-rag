"""Embedding wrapper that adds nomic-embed-text task prefixes.

nomic-embed-text is trained with asymmetric prefixes: passages are embedded as
``search_document: <text>`` and queries as ``search_query: <text>``. LangChain's
``OllamaEmbeddings`` sends raw text, so this subclass adds the prefixes. The
same object is used at ingest time and at query time, so the two sides always
agree.
"""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from cerulean_rag.config import Settings, get_settings

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """``OllamaEmbeddings`` with nomic task prefixes on documents and queries."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(QUERY_PREFIX + text)


def get_embeddings(settings: Settings | None = None) -> PrefixedOllamaEmbeddings:
    """Embeddings client configured from settings (model + Ollama URL)."""
    s = settings or get_settings()
    return PrefixedOllamaEmbeddings(model=s.EMBED_MODEL, base_url=s.OLLAMA_BASE_URL)
