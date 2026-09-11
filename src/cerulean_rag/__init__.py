"""Cerulean Systems RAG assistant.

A retrieval-augmented generation pipeline over the 13-document Cerulean
Systems corpus. Generation and embeddings run locally through Ollama.

Module map (filled in step by step):
    config          settings from environment / .env
    logging_setup   console + file logging
    models          shared data types (documents, chunks, answers)
    loaders         PDF -> text + metadata
    chunking        section-aware chunking
    security        injection scanner, input guard, output checks
    embeddings      prefixed nomic-embed-text wrapper
    ingest          build the vector store + chunk file
    retrieval       hybrid search, fusion, sub-queries, metadata notes
    prompts         system prompt and templates
    generation      structured LLM call
    pipeline        ask(question) -> AnswerResult
    cli             `ask` and `chat` commands
"""

__version__ = "0.1.0"
