# Cerulean Systems RAG Assistant

A grounded retrieval-augmented assistant over the 13-document Cerulean Systems
corpus, running entirely on local open-weight models through Ollama
(qwen2.5:7b-instruct for generation, nomic-embed-text for embeddings).

- **[SETUP.md](SETUP.md)** — how to install and run everything on a fresh machine, with measured timings.
- **[PROJECT_LOG.md](PROJECT_LOG.md)** — in-depth account of how it was built, every design decision and its reason, evaluation results and known limitations.
- **[NOTES.md](NOTES.md)** — raw working notes and measurements per build step.
- **[eval/results/](eval/results/)** — committed evaluation runs (12 official questions + 10 extras).

Current evaluation status (qwen2.5:7b-instruct, CPU-only laptop): 11 of the
12 official questions pass; all five safety/hallucination questions pass;
the one failure (a multi-document date-range calculation) is a documented
reasoning limit of the 7B model.

The full README (architecture diagram, technology rationale, weaknesses,
deliberate omissions) is the final build step and is not yet written.
