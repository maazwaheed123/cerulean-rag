# Cerulean Systems RAG Assistant

A grounded retrieval-augmented assistant over the 13-document Cerulean Systems
corpus, running entirely on local open-weight models through Ollama
(qwen2.5:7b-instruct for generation, nomic-embed-text for embeddings).

- **[SETUP.md](SETUP.md)** — install, run, and point the assistant at your own documents. Start here.
- **[NOTES.md](NOTES.md)** — working notes, design decisions and measurements from the build.
- **[eval/results/](eval/results/)** — committed evaluation runs (12 official questions + 10 extras).

Current evaluation status (qwen2.5:7b-instruct, CPU-only laptop): 11 of the
12 official questions pass; all five safety/hallucination questions pass;
the one failure (a multi-document date-range calculation) is a documented
reasoning limit of the 7B model.

The full README (architecture diagram, technology rationale, weaknesses,
deliberate omissions) is the final build step and is not yet written.
