# NOTES.md — scratch notes carried into the README (Step 11)

Working notes recorded during the build. Every number here was measured on the
target machine, not estimated. Sections marked TODO are filled in later steps.

## Target machine (measured 11 Sep 2026)

| Item | Value |
|---|---|
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz, 4 cores / 8 threads |
| GPU | Intel Iris Xe (integrated) — no CUDA GPU, all inference is CPU |
| RAM | 19.8 GB |
| OS | Windows 11 Pro 10.0.26200 |
| Shell | PowerShell 5.1 |
| git | 2.52.0.windows.1 |
| Pre-existing Python | 3.14.4 (too new for the ML stack; left in place) |

## Step 0 — environment setup (commands run verbatim)

### Python 3.12 (installed alongside 3.14, user scope)

```powershell
winget install -e --id Python.Python.3.12 --scope user
```

Result: `Python.Python.3.12` version **3.12.10** installed to
`C:\Users\maazw\AppData\Local\Programs\Python\Python312\python.exe`.
The `py` launcher picks it up as `py -3.12` (3.14 stays the default `py`).
A new shell is required after install so PATH refreshes.

```text
> py -3.12 --version
Python 3.12.10
> py -0p
 -V:3.14 *        C:\Users\maazw\AppData\Local\Python\pythoncore-3.14-64\python.exe
 -V:3.12          C:\Users\maazw\AppData\Local\Programs\Python\Python312\python.exe
```

### Ollama for Windows

```powershell
winget install -e --id Ollama.Ollama
```

Result: **Ollama 0.34.0** installed to
`C:\Users\maazw\AppData\Local\Programs\Ollama\ollama.exe`. The installer
also starts the tray app (`ollama app.exe`) and the server (`ollama.exe`).
Open a NEW shell after install; existing shells do not see the updated PATH
(this bit us: `ollama` was "not recognized" until the shell was restarted).

```text
> ollama --version
ollama version is 0.34.0
> curl http://localhost:11434
Ollama is running
```

### Models

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
# optional, for the Step 10 A/B comparison
ollama pull qwen2.5:3b-instruct
```

| Model | Size on disk | Pull time (measured) | Notes |
|---|---|---|---|
| `nomic-embed-text:latest` (id 0a109f422b47) | 274 MB | 485.8 s | pulled concurrently with the 7B model at ~0.5 MB/s |
| `qwen2.5:7b-instruct` (id 845dbda0ea48) | 4.7 GB | 1203.6 s (~20 min) | download ran at ~2.5 MB/s on this connection |
| `qwen2.5:3b-instruct` (id 357c53fb659c, optional, Step 10) | 1.9 GB | 557.4 s (~9 min) | pulled after the two required models, ~3.1 MB/s |

Download time is dominated by the network here, not the machine; on a fast
connection the 7B pull is a few minutes.

### Smoke tests

Generation check exactly as in the plan:

```text
> ollama run qwen2.5:7b-instruct --verbose "Reply with exactly: OK"
OK

total duration:       16.5027102s
load duration:        14.3550605s
prompt eval count:    34 token(s)
prompt eval duration: 1.859514s
prompt eval rate:     18.28 tokens/s
eval count:           2 token(s)
eval duration:        258.916999ms
eval rate:            7.72 tokens/s
```

The 7.72 tok/s figure is from only 2 output tokens, so two longer runs were
made to get numbers worth putting in the README:

| Run | Prompt tokens | Prompt eval rate | Output tokens | Generation rate |
|---|---|---|---|---|
| `ollama run` "five short sentences" | 41 (24 cached) | 17.28 tok/s | 69 | **3.47 tok/s** |
| `/api/generate`, ~1.8k-token prompt, `num_ctx=8192`, `num_predict=5` | 1847 | **17.59 tok/s** (104.98 s) | 2 | 6.52 tok/s |

Cold model load (first call after pull): 14.4 s. Warm load: ~5 ms.

Same two runs on the 3B development model, for the Step 10 comparison:

| Run | Prompt tokens | Prompt eval rate | Output tokens | Generation rate |
|---|---|---|---|---|
| `ollama run qwen2.5:3b-instruct` "five short sentences" | 41 | 49.24 tok/s | 97 | **8.37 tok/s** |
| `/api/generate` on 3B, same ~1.8k-token prompt | 1847 | **45.02 tok/s** (41.03 s) | 2 | 13.53 tok/s |

3B cold load: 3.2 s. Roughly 2.5x faster than 7B on both prompt processing
and generation, so a full RAG question is ~1.5 min on 3B vs ~4 min on 7B.

**Design implication (7B on this CPU).** Generation runs at roughly 3.5 tok/s
(slower than the 4–7 tok/s the plan assumed), and prompt processing at
roughly 17.5 tok/s. For a RAG call with a ~900-token system prompt plus
8 chunks of ~250 tokens (~3,000 tokens in) and a ~300-token JSON answer out,
that is about 170 s of prompt processing + 85 s of generation, i.e. roughly
4 minutes per question, 50 minutes for the 12-question eval on the 7B model.
Consequences carried into later steps:
- Keep ONE LLM call per question (already the plan's rule); every extra call
  is minutes, not seconds.
- Prompt size matters as much as answer length. Keep the system prompt tight,
  keep chunks ~250 tokens, and consider TOP_K=6 if recall allows (Step 9).
- Ollama reuses the KV cache for an unchanged prompt prefix on the same
  loaded model ("prompt eval cached" above). Keeping the system prompt as an
  unchanging prefix lets consecutive questions skip re-processing it.
- Use `qwen2.5:3b-instruct` for iteration and run the FINAL eval on 7B.
- Keep Ollama's default keep-alive (5 min) or raise it during the eval so the
  4.7 GB model is not reloaded between questions.

Embedding check (PowerShell equivalent of the curl command in the plan):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:11434/api/embed `
  -ContentType "application/json" `
  -Body '{"model":"nomic-embed-text","input":"search_query: annual leave"}'
```

Result: `model=nomic-embed-text`, 1 vector, **dim=768**, first values
`0.0251, 0.0061, -0.1799`. First call took 2948 ms including model load; the
model stays resident afterwards so subsequent calls are much faster.

## Source material read in Step 0

- `SAITC_Applied_AI_Engineer_Assignment.pdf` (5 pages) — brief, 12 test
  questions, assessment areas, deliverables. Deadline stated in the brief is
  31 Aug 2026; the as-of date for "current" questions is 27 Aug 2026.
- `corpus/00_README_Corpus_Overview.pdf` (2 pages) — 13 text-based PDFs,
  metadata block on page 1 of each plus `corpus_manifest.json`; corpus
  contradicts itself deliberately; some docs carry text addressed to an AI
  assistant; tables appear in most documents; currency written "SAR 4,500".
  This README PDF is assignment material, not company knowledge, so it is
  kept out of the retrieval corpus (see Step 1).

## Resolutions / gotchas (for the README "known limitations" and setup notes)

- PyMuPDF (`fitz`) fails to import with `DLL load failed ... The filename or
  extension is too long` when installed under a very deep directory path on
  Windows. Keep the project venv at a short path such as
  `C:\Users\<user>\Desktop\cerulean-rag\.venv`.
- `pip 25.0.1` ships with Python 3.12.10; upgrade pip inside the venv in
  Step 1 (`python -m pip install --upgrade pip`).

## Step 1 — scaffold, config, logging, corpus copy

Commands run (verbatim):

```powershell
git init -b main
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip      # 25.0.1 -> 26.2.1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt  # 102.6 s, 230 packages
.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Resolved versions pinned into `requirements.txt` (full transitive freeze in
`requirements.lock.txt`):

| Package | Version | | Package | Version |
|---|---|---|---|---|
| langchain | 1.4.0 | | chromadb | 1.5.9 |
| langchain-core | 1.6.2 | | pymupdf | 1.28.2 |
| langchain-community | 0.4.2 | | rank-bm25 | 0.2.2 |
| langchain-ollama | 1.1.0 | | pydantic | 2.13.5 |
| langchain-chroma | 1.1.0 | | pydantic-settings | 2.15.0 |
| langchain-text-splitters | 1.1.2 | | python-dotenv | 1.2.3 |
| PyYAML | 6.0.3 | | rich | 15.0.0 |
| pytest | 9.1.1 | | | |

Notes:
- LangChain resolved to the 1.x line. The plan was written against the
  classic API names (`ChatOllama`, `OllamaEmbeddings`, `Chroma`,
  `RecursiveCharacterTextSplitter`); these still exist in the partner
  packages, but check each import against the installed version in the step
  that uses it rather than assuming 0.x signatures.
- Windows PowerShell 5.1 `Out-File` writes a UTF-8 BOM. A BOM at the top of
  `requirements.txt` breaks setuptools' dynamic-dependencies reader, so the
  files were re-saved without BOM. The README should tell reviewers to edit
  requirement files with a BOM-free editor, or to use `pip freeze |
  Set-Content -Encoding ascii` style commands.
- `00_README_Corpus_Overview.pdf` and the assignment brief live under `docs/`,
  not `corpus/`, so they are never embedded or retrieved.
- Package layout uses `pyproject.toml` with `package-dir = {"" = "src"}` and
  `pip install -e .`, so `python -m cerulean_rag.cli` works from any directory
  once the venv is active.
- PyMuPDF 1.28 prints `warning: The fitz API is deprecated ... Use import
  pymupdf instead` on `import fitz`. Use `import pymupdf` in `loaders.py`
  (Step 2). The plan's VERIFY line for Step 1 still uses `fitz` and passes.
- git on this machine has `core.autocrlf=true`, and it classified the small
  text-based corpus PDFs as text ("LF will be replaced by CRLF"). Checking
  them out on another machine would have rewritten bytes inside the PDFs.
  Fixed with `.gitattributes`: `* text=auto eol=lf` plus `*.pdf binary`.
  The copied corpus was SHA-256 compared against the original pack before
  the first commit.
