# Setup from scratch (Windows 11, no GPU required)

These are the exact steps used to build and run this project on a fresh
machine. Every command was run on the development laptop (Intel i5-1135G7,
19.8 GB RAM, no NVIDIA GPU, Windows 11 Pro). A machine with an NVIDIA GPU
runs the same steps and is 15-25x faster at the model stage.

Times quoted are from that laptop.

## 1. Prerequisites

| Tool | Version used | Install |
|---|---|---|
| Python | 3.12.10 (3.13/3.14 are too new for the ML stack) | `winget install -e --id Python.Python.3.12 --scope user` |
| Ollama | 0.34.0 | `winget install -e --id Ollama.Ollama` |
| git | 2.52 | `winget install -e --id Git.Git` (if missing) |

Open a **new** PowerShell after the installs so PATH refreshes, then confirm:

```powershell
py -3.12 --version          # Python 3.12.10
ollama --version            # ollama version is 0.34.0
curl http://localhost:11434 # Ollama is running
```

Ollama runs as a tray app on Windows. If `curl` fails, start it with
`ollama serve` or open the Ollama app.

## 2. Pull the models

```powershell
ollama pull qwen2.5:7b-instruct     # generation, 4.7 GB  (~20 min at 2.5 MB/s)
ollama pull nomic-embed-text        # embeddings, 274 MB
ollama pull qwen2.5:3b-instruct     # optional: faster model for iteration / comparison, 1.9 GB
```

Check: `ollama list` shows all three.

Smoke test (records your machine's speed for the README):

```powershell
ollama run qwen2.5:7b-instruct --verbose "Reply with exactly: OK"
```

Look at `eval rate` (tokens/s). This laptop: ~3.5 tok/s generation,
~17.5 tok/s prompt processing. A 3,000-token RAG prompt plus a 250-token
answer therefore takes 2.5-4 minutes here; on an RTX 4050 expect ~10-15 s.

## 3. Get the code and create the environment

```powershell
git clone <this repository URL> cerulean-rag
cd cerulean-rag
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt      # ~100 s, pinned versions
pip install -e .                     # installs the cerulean_rag package from src/
```

Keep the project at a short path such as `C:\Users\<you>\Desktop\cerulean-rag`.
PyMuPDF fails to load from very deep directory paths on Windows.

Verify:

```powershell
python -c "import cerulean_rag, langchain, chromadb, pymupdf, rank_bm25; print('imports ok')"
python -m pytest -q -m "not ollama"   # 98 offline tests, ~15 s
```

## 4. Configuration

All settings have working defaults in `src/cerulean_rag/config.py`. To change
any, copy `.env.example` to `.env` and edit. The keys that matter most:

| Key | Default | Meaning |
|---|---|---|
| `GEN_MODEL` | `qwen2.5:7b-instruct` | generation model tag in Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `AS_OF_DATE` | `2026-08-27` | the date "current" refers to (assignment as-of date; taken from config, not the clock, for reproducibility) |
| `TOP_K` | `8` | excerpts sent to the model |
| `SIM_THRESHOLD` | `0.65` | below this the model is told retrieval confidence is LOW |
| `NUM_CTX` | `8192` | context window; lower to 6144 if a 6 GB GPU offloads layers |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |

## 5. Ingest the corpus (once)

```powershell
python scripts/ingest.py
```

Expected output (this laptop, 25 s):

```text
13 documents, 88 chunks, 3 flagged for embedded instructions, embed 19.1 s, total 24.9 s
```

This writes `data/chroma/` (vector store) and `data/chunks.jsonl` (chunk
table used for keyword search and audit). Both are git-ignored and rebuilt
from scratch on every run. `--no-embed` writes only the chunk table (no Ollama
needed).

## 6. Ask questions

One question:

```powershell
python scripts/ask.py "How much notice must an employee give when resigning during probation?"
```

Useful flags: `--show-context` (list the retrieved passages and scores),
`--json` (full machine-readable result), `--model qwen2.5:3b-instruct`
(faster, weaker).

Interactive:

```powershell
python scripts/chat.py
```

Commands inside the chat: `/help`, `/context`, `/model <tag>`, `/json`,
`/exit`. Each question is answered independently (no conversation memory,
by design).

Debug retrieval without the model:

```powershell
python scripts/retrieval_check.py "What is the limit?"
```

## 7. Run the evaluation

```powershell
python scripts/run_eval.py                       # all 22 questions (~55 min here, ~5 min on a GPU)
python scripts/run_eval.py --only Q1,Q2,Q3       # a subset
python scripts/run_eval.py --model qwen2.5:3b-instruct --tag 3b
python scripts/run_eval.py --repeat 2            # stability check
```

Results land in `eval/results/<timestamp>_<model>[_tag].md` and `.json`.
Existing runs are committed there.

## 8. Logs

- `logs/app.log` – application log with full tracebacks (git-ignored)
- `logs/queries.jsonl` – one JSON line per question: retrieved chunks and
  scores, decision, citations, warnings, timings, token counts, prompt hash

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama at http://localhost:11434` | start Ollama (`ollama serve` or the tray app) |
| `The knowledge base has not been built yet` | run `python scripts/ingest.py` |
| `Model 'x' is not available in Ollama` | `ollama pull <tag>` |
| `DLL load failed ... filename or extension is too long` on `import pymupdf` | move the project to a shorter path |
| Requirements install error mentioning an invalid requirement at line 1 | the file has a UTF-8 BOM (PowerShell `Out-File`); re-save without BOM |
| Slow answers | expected on CPU; use `--model qwen2.5:3b-instruct` for iteration, or a GPU machine |

## 10. Run the tests

```powershell
python -m pytest -q                 # everything; Ollama-dependent tests skip if the server is down
python -m pytest -q -m "not ollama" # offline only
python -m pytest -q -m ollama       # live retrieval recall + two end-to-end questions (~5 min here)
```
