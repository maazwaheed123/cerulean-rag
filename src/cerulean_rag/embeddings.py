"""nomic-embed-text is trained with asymmetric prefixes (search_document: for
passages, search_query: for questions) and LangChain sends raw text, so the
prefixes are added here. One class for both sides keeps them in step.
"""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from cerulean_rag.config import Settings, get_settings

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(QUERY_PREFIX + text)


def get_embeddings(settings: Settings | None = None) -> PrefixedOllamaEmbeddings:
    s = settings or get_settings()
    return PrefixedOllamaEmbeddings(model=s.EMBED_MODEL, base_url=s.OLLAMA_BASE_URL)
