# Setup and usage

Everything you need to install this, run it against the bundled corpus, and
point it at your own documents.

It runs entirely on your machine: generation and embeddings go through Ollama
with open-weight models, and nothing is sent to a hosted API.

- [1. Prerequisites](#1-prerequisites)
- [2. Pull the models](#2-pull-the-models)
- [3. Install](#3-install)
- [4. Configuration](#4-configuration)
- [5. Build the index](#5-build-the-index)
- [6. Ask questions](#6-ask-questions)
- [7. **Use your own documents**](#7-use-your-own-documents)
- [8. Evaluate on your own questions](#8-evaluate-on-your-own-questions)
- [9. Logs](#9-logs)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Tests](#11-tests)

Commands are written for PowerShell on Windows 11. On macOS or Linux the only
differences are `python3` instead of `python` and `source .venv/bin/activate`
instead of the `Activate.ps1` script.

Timings quoted throughout are from the development laptop: Intel i5-1135G7
(4 cores / 8 threads), 19.8 GB RAM, **no GPU**, Windows 11 Pro. A machine with
an NVIDIA GPU runs the same steps 15-25x faster at the model stage.

---

## 1. Prerequisites

| Tool | Version used | Install |
|---|---|---|
| Python | 3.12.10 (3.13 and 3.14 are too new for the ML stack) | `winget install -e --id Python.Python.3.12 --scope user` |
| Ollama | 0.34.0 | `winget install -e --id Ollama.Ollama` |
| git | 2.52 | `winget install -e --id Git.Git` |

Open a **new** PowerShell after installing so PATH refreshes, then confirm:

```powershell
py -3.12 --version
ollama --version
curl http://localhost:11434
```

Ollama runs as a tray app on Windows. If `curl` fails, start it with
`ollama serve` or open the Ollama app.

## 2. Pull the models

```powershell
ollama pull qwen2.5:7b-instruct     # generation, 4.7 GB  (~20 min at 2.5 MB/s)
ollama pull nomic-embed-text        # embeddings, 274 MB
ollama pull qwen2.5:3b-instruct     # optional: faster model for iteration
```

`ollama list` should show all three.

To record your own machine's speed:

```powershell
ollama run qwen2.5:7b-instruct --verbose "Reply with exactly: OK"
```

Look at `eval rate`. This laptop manages ~3.5 tok/s generation and ~17.5 tok/s
prompt processing, so a question costs 3-6 minutes. On an RTX 4050 expect
10-15 seconds.

## 3. Install

```powershell
git clone <this repository URL> cerulean-rag
cd cerulean-rag
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

> Keep the project at a short path such as `C:\Users\<you>\Desktop\cerulean-rag`.
> PyMuPDF fails to load from very deep directory paths on Windows.

Verify:

```powershell
python -c "import cerulean_rag, langchain, chromadb, pymupdf, rank_bm25; print('imports ok')"
python -m pytest -q -m "not ollama"
```

You should see `121 passed` in about 35 seconds.

## 4. Configuration

Every setting has a working default in `src/cerulean_rag/config.py`. To change
one, copy `.env.example` to `.env` and edit it, or set an environment variable
of the same name.

| Key | Default | Meaning |
|---|---|---|
| `CORPUS_DIR` | `./corpus` | folder holding your documents and their manifest |
| `CHROMA_DIR` | `./data/chroma` | where the vector index is written |
| `CHUNKS_FILE` | `./data/chunks.jsonl` | chunk table used for keyword search and audit |
| `COLLECTION_NAME` | `cerulean_docs` | Chroma collection name |
| `AS_OF_DATE` | `2026-08-27` | the date "current" means; read from config, never the system clock, so results are reproducible |
| `GEN_MODEL` | `qwen2.5:7b-instruct` | generation model tag in Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `TOP_K` | `8` | excerpts sent to the model |
| `SIM_THRESHOLD` | `0.65` | below this, the model is told retrieval confidence is LOW |
| `NUM_CTX` | `8192` | context window; lower to 6144 if a 6 GB GPU offloads layers |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |

**If you bring your own documents, set `AS_OF_DATE` to today's date** (or to
whatever date you want "current" to mean), otherwise the assistant will reason
about which of your documents are in force using 27 August 2026.

## 5. Build the index

```powershell
python scripts/ingest.py
```

```text
13 documents, 88 chunks, 3 flagged for embedded instructions, embed 19.1 s, total 24.9 s
```

This writes `data/chroma/` and `data/chunks.jsonl`. Both are git-ignored and
rebuilt from scratch every run — there is no incremental mode, so **re-run this
whenever you add, remove or edit a document**. `--no-embed` writes only the
chunk table and needs no Ollama, which is useful for checking that your
documents parse before you spend time embedding them.

## 6. Ask questions

```powershell
python scripts/ask.py "How much notice must an employee give when resigning during probation?"
```

Flags: `--show-context` lists the retrieved passages with their scores and the
signals the system gave the model, `--json` prints the full machine-readable
result, `--model qwen2.5:3b-instruct` switches model for one run.

Interactive:

```powershell
python scripts/chat.py
```

Inside the chat: `/help`, `/context`, `/model <tag>`, `/json`, `/exit`. Each
question is answered independently — there is deliberately no conversation
memory, so an instruction planted in one document cannot carry over into a
later turn.

To inspect retrieval without paying for generation:

```powershell
python scripts/retrieval_check.py "What is the limit?"
```

---

## 7. Use your own documents

The assistant is not tied to the bundled corpus. It needs two things: a folder
of documents, and a `corpus_manifest.json` in that folder describing them.

### 7.1 The manifest

`corpus_manifest.json` is the authority for document metadata. Nothing is
guessed from filenames.

```json
{
  "documents": [
    {
      "file": "remote_working_policy.pdf",
      "document_id": "OPS-POL-001",
      "title": "Remote Working Policy",
      "version": "1.0",
      "effective_date": "2026-02-01",
      "owner": "People Operations",
      "classification": "Internal"
    }
  ]
}
```

| Field | Required | Notes |
|---|---|---|
| `file` | yes | filename, relative to the corpus folder |
| `document_id` | yes | your identifier; this is what the assistant cites, so make it recognisable |
| `title` | yes | shown in citations and in the passage header |
| `effective_date` | yes | `2026-02-01` or `1 February 2026` |
| `version` | no | recommended; appears in citations |
| `owner` | no | |
| `classification` | no | |
| `supersedes` | no | see below |

If a required field is missing you get a clear error naming the document and
the field, rather than a stack trace.

### 7.2 Marking a document as superseded

Give the newer document a `supersedes` value whose **first token is the older
document's `document_id`**:

```json
{ "document_id": "SALES-PL-2026", "supersedes": "SALES-PL-2025 v1.0", ... }
```

The chain is derived across the whole corpus at load time. The older document
is then marked historical, ranked slightly below current documents, and — this
is the point — when both turn up for the same question, the system tells the
model that they may disagree and that it must report the conflict rather than
quietly pick one. A `supersedes` value pointing at something not in your corpus
is recorded as a note instead ("supersedes X, which is not in the knowledge
base"), which is what makes the assistant able to say a referenced document is
missing.

### 7.3 File formats and how they are chunked

**PDF is the best-supported format.** Tables are detected and rendered as pipe
rows, so `| Professional | SAR 5,200 | 50 users |` stays on one line and a
price keeps its plan. Plain text extraction would put each cell on its own
line and lose that.

`.txt` and `.md` also load, but PyMuPDF reflows them: a question and its answer
end up on one line, and Markdown syntax is not interpreted. They work fine for
prose; use PDF when structure matters.

Chunking picks a strategy per document, in this order:

1. **Two or more numbered headings** (`1. Purpose`, `4.2 Accrual`) → one chunk
   per section, tables kept whole with their heading. **This gives the best
   retrieval and the most precise citations.**
2. Otherwise, **three or more lines ending in `?`** → one chunk per question
   and its answer, for FAQ-style documents.
3. Otherwise → a plain character splitter with overlap.

So if you control the source documents, numbering the sections is the single
highest-value thing you can do.

Every chunk is embedded behind a context header carrying its identity:

```text
[OPS-POL-001 v1.0 | Remote Working Policy | effective 2026-02-01 | current | 2. Eligibility]
```

### 7.4 Keeping corpora side by side

Point the four path settings somewhere new and the bundled corpus stays intact:

```powershell
$env:CORPUS_DIR      = "./mycorpus"
$env:CHROMA_DIR      = "./mydata/chroma"
$env:CHUNKS_FILE     = "./mydata/chunks.jsonl"
$env:COLLECTION_NAME = "my_docs"
```

Or put the same keys in `.env` to make the switch permanent.

### 7.5 Worked example, start to finish

This is a real run, reproduced verbatim. It uses three documents in three
different formats to show that they all work.

**Create the folder and drop the documents in:**

```powershell
mkdir mycorpus
```

- `mycorpus\remote.pdf` — a PDF with numbered sections `1. Purpose`,
  `2. Eligibility`, `3. Equipment`
- `mycorpus\expenses.txt` — sections `1. Booking` and `2. Limits`, the latter
  reading "Hotel spend is capped at USD 180 per night."
- `mycorpus\faq.md` — three question-and-answer pairs

**Write `mycorpus\corpus_manifest.json`:**

```json
{
  "documents": [
    {
      "file": "remote.pdf",
      "document_id": "OPS-POL-001",
      "title": "Remote Working Policy",
      "version": "1.0",
      "effective_date": "2026-02-01",
      "owner": "People Operations",
      "classification": "Internal"
    },
    {
      "file": "expenses.txt",
      "document_id": "FIN-POL-010",
      "title": "Travel Expense Rules",
      "version": "2.0",
      "effective_date": "2026-03-01",
      "owner": "Finance",
      "classification": "Internal"
    },
    {
      "file": "faq.md",
      "document_id": "SUP-FAQ-020",
      "title": "Support FAQ",
      "version": "1.0",
      "effective_date": "2026-01-15",
      "owner": "Support",
      "classification": "Public"
    }
  ]
}
```

**Point the settings at it:**

```powershell
$env:CORPUS_DIR      = "./mycorpus"
$env:CHROMA_DIR      = "./mydata/chroma"
$env:CHUNKS_FILE     = "./mydata/chunks.jsonl"
$env:COLLECTION_NAME = "my_docs"
$env:AS_OF_DATE      = "2026-09-12"
```

**Check the documents parse before embedding anything:**

```powershell
python scripts/ingest.py --no-embed
```

**Build the index:**

```powershell
python scripts/ingest.py
```

```text
3 documents, 8 chunks, 0 flagged for embedded instructions, embed 3.4 s, total 3.9 s
```

**Confirm retrieval finds the right passage, without waiting for the model:**

```powershell
python scripts/retrieval_check.py "hotel spend limit"
```

**Ask:**

```powershell
python scripts/ask.py "What is the hotel spend limit per night?"
```

```text
┌─ ANSWER  ·  confidence high ────────────────────────────────────────────────┐
│                                                                             │
│  The hotel spend limit per night is USD 180.                                │
│                                                                             │
└─ answered from the retrieved documents ─────────────────────────────────────┘

Where this comes from
  1. FIN-POL-010  Travel Expense Rules
     section 2. Limits  ·  version 2.0, effective 2026-03-01
     "Hotel spend is capped at USD 180 per night."

retrieval 3.3 s / generation 78.0 s / model qwen2.5:7b-instruct / 8 passages used
```

The citation names the document, the section and the sentence the figure came
from, so every answer can be checked against the source.

### 7.6 What to expect from your own documents

- **Absent facts.** If your documents do not contain the answer, the assistant
  says so rather than inventing one. That is the intended behaviour, not a
  failure.
- **Conflicts.** Two documents giving different values for the same thing are
  reported together with the reasoning for which one wins, provided both are
  retrieved. The supersedes chain and effective dates drive that decision.
- **Planted instructions.** Every document is scanned at ingest for text
  addressed to an AI assistant ("ignore previous instructions", "SYSTEM:",
  HTML comments aimed at models). Such text is indexed as ordinary content but
  flagged, never obeyed, and never used as a citation. The ingest line tells
  you how many chunks were flagged; on a clean corpus that is `0`.
- **Arithmetic.** Multi-step calculations across documents are the weakest
  area of a 7B model. The system supplies correct calendar arithmetic for
  date-range questions, but check any answer that multiplies or sums.

---

## 8. Evaluate on your own questions

`eval/questions.yaml` holds the question set. Each entry is scored generically —
nothing in the pipeline ever reads this file.

```yaml
- id: Q1
  category: direct
  question: "What is the hotel spend limit per night?"
  expected_decision: [answer]          # answer | insufficient_evidence | needs_clarification | conflict_resolved | refused
  expected_docs: [FIN-POL-010]         # every id listed must appear in the citations
  must_contain: ["USD 180"]            # case-insensitive regexes, all must match
  must_not_contain: []                 # regexes that must NOT appear in the answer
  notes: "Direct grounded answer."
```

Optional keys: `expected_conflict: true`, `expected_injection_noticed: true`,
`min_clarification_options: 2`.

```powershell
python scripts/run_eval.py                     # whole set
python scripts/run_eval.py --only Q1,Q2        # a subset
python scripts/run_eval.py --repeat 2          # stability across repeats
python scripts/run_eval.py --questions eval/my_questions.yaml --tag mine
```

Results are written to `eval/results/<timestamp>_<model>[_tag].md` and `.json`.

> Single runs are noisy. On CPU, Ollama at temperature 0 with a fixed seed is
> still not reproducible run to run, so judge a change over repeated runs
> rather than one.

## 9. Logs

- `logs/app.log` — application log with full tracebacks
- `logs/queries.jsonl` — one JSON line per question: retrieved chunks and their
  scores, decision, citations, warnings, timings, token counts, prompt hash

Both are git-ignored. `queries.jsonl` is the audit trail: every answer can be
traced back to the passages and signals that produced it.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama at http://localhost:11434` | start Ollama (`ollama serve` or the tray app) |
| `The knowledge base has not been built yet` | run `python scripts/ingest.py` |
| `Model 'x' is not available in Ollama` | `ollama pull <tag>` |
| `Manifest not found: ...corpus_manifest.json` | your `CORPUS_DIR` has no manifest, or points at the wrong folder |
| `manifest entry for X is missing required field(s): ...` | add the named fields to that entry |
| `X: no extractable text in Y.pdf` | the PDF is a scan; OCR it first, as there is no OCR in this pipeline |
| Answers cite the wrong section, or sections look merged | the document has no numbered headings, so the fallback splitter was used — see §7.3 |
| Assistant treats an old document as current | set `AS_OF_DATE`, and add `supersedes` to the newer document |
| `DLL load failed ... filename or extension is too long` on `import pymupdf` | move the project to a shorter path |
| Requirements install error about an invalid requirement at line 1 | the file has a UTF-8 BOM; re-save without one |
| Answers are very slow | expected on CPU; use `--model qwen2.5:3b-instruct` to iterate, or run on a GPU |

## 11. Tests

```powershell
python -m pytest -q                  # everything; Ollama tests skip if the server is down
python -m pytest -q -m "not ollama"  # 121 offline tests, ~35 s
python -m pytest -q -m ollama        # 10 live tests: retrieval recall, end-to-end questions (~2 min)
```
