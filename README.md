# Cerulean Systems RAG Assistant

A grounded question-answering assistant over the 13-document Cerulean Systems
corpus. It runs entirely on local open-weight models through Ollama —
`qwen2.5:7b-instruct` for generation, `nomic-embed-text` for embeddings — with
no hosted API in the path.

The interesting part of this corpus is not the happy path. Two documents
disagree about the price of the same plan, one is superseded by another, one is
overdue for review, three contain text planted to hijack an AI assistant, and
several plausible questions have no answer anywhere in the corpus. The design
below is mostly about those cases.

**Current status:** 11 of the 12 assessment questions pass; all five
safety and hallucination questions pass; the single failure is a multi-document
date arithmetic question, which is a documented limit of a 7B model on CPU.

- [Quick start](#quick-start)
- [Hardware and timings](#hardware-and-timings)
- [Architecture](#architecture)
- [Technology choices](#technology-choices)
- [How the hard cases are handled](#how-the-hard-cases-are-handled)
- [Assumptions](#assumptions)
- [Known limitations](#known-limitations)
- [The five weaknesses that concern me most](#the-five-weaknesses-that-concern-me-most)
- [What I deliberately did not build](#what-i-deliberately-did-not-build)
- [Evaluation](#evaluation)
- [Repository map](#repository-map)
- [Closing thought](#closing-thought)

> **[SETUP.md](SETUP.md)** covers the same install in more depth, plus how to
> point the assistant at **your own documents**.
> **[NOTES.md](NOTES.md)** holds the working notes and measurements per build step.

---

## Quick start

Written for PowerShell on Windows 11. On macOS or Linux use `python3` and
`source .venv/bin/activate`; nothing else differs.

### 1. Prerequisites

| Tool | Version used | Install |
|---|---|---|
| Python | **3.12.10** — 3.13 and 3.14 are too new for this stack | `winget install -e --id Python.Python.3.12 --scope user` |
| Ollama | **0.34.0** | `winget install -e --id Ollama.Ollama` |
| git | 2.52 | `winget install -e --id Git.Git` |

Open a **new** terminal after installing so PATH refreshes, then check all
three respond:

```powershell
py -3.12 --version
ollama --version
curl http://localhost:11434
```

Ollama runs as a tray app on Windows. If `curl` fails, start it with
`ollama serve` or open the Ollama app.

### 2. Pull the models

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

4.7 GB and 274 MB. The download took about 20 minutes on a 2.5 MB/s line.
`ollama pull qwen2.5:3b-instruct` is optional — it is roughly 2.5x faster and
useful for iterating, at some cost in accuracy.

### 3. Install the project

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
> PyMuPDF fails to load its DLL from very deep paths on Windows.

Check it imports and the offline tests pass (about 20 seconds):

```powershell
python -c "import cerulean_rag, langchain, chromadb, pymupdf, rank_bm25; print('imports ok')"
python -m pytest -q -m "not ollama"
```

### 4. Configuration

**Nothing is required.** Every setting has a working default in
`src/cerulean_rag/config.py`, and the corpus is already in `corpus/`. To change
anything, copy `.env.example` to `.env` and edit it. The settings that matter:

| Key | Default | Meaning |
|---|---|---|
| `AS_OF_DATE` | `2026-08-27` | the date "current" refers to |
| `GEN_MODEL` | `qwen2.5:7b-instruct` | generation model tag in Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `TOP_K` | `8` | excerpts sent to the model |
| `SIM_THRESHOLD` | `0.65` | below this the model is told retrieval confidence is LOW |
| `NUM_CTX` | `8192` | context window |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |

`AS_OF_DATE` is read from configuration and **never from the system clock**.
The assignment's "current" means 27 August 2026, and a question about which
price is in force must give the same answer next year as it does today.

### 5. Ingest the corpus

```powershell
python scripts/ingest.py
```

```text
13 documents, 88 chunks, 3 flagged for embedded instructions, embed 19.1 s, total 24.9 s
```

Three flagged chunks is correct — those are the planted injections, which are
indexed as ordinary content but tagged. The build is always a full rebuild;
re-run it after changing any document.

### 6. Ask a question

```powershell
python scripts/ask.py "What is the current price of the Atlas Professional plan?"
```

```text
┌─ CONFLICT RESOLVED  ·  confidence high ────────────────────────────────────────────────────────┐
│                                                                                                │
│  The current price of the Atlas Professional plan is SAR 5,200 per month for up to 50 users,   │
│  billed annually in advance.                                                                   │
│                                                                                                │
└─ the documents disagreed; the conflict is set out below ───────────────────────────────────────┘

Why the documents disagreed
  Atlas Professional plan price
    SALES-PL-2026  SAR 5,200 per month for up to 50 users
    SUP-FAQ-001    SAR 4,500 per month for up to 50 users
    RESOLVED       SALES-PL-2026 is the current document and its value is preferred.
    because        SALES-PL-2026 is the current, superseding document, and it provides the current
                   price. SUP-FAQ-001 is marked as overdue and weaker evidence.

Where this comes from
  1. SALES-PL-2026  Atlas Platform Price List
     section 1. Subscription plans  ·  version 2.0, effective 2026-03-01
     "SAR 5,200 · Up to 50 · 500 GB"

retrieval 3.6 s / generation 204.2 s / model qwen2.5:7b-instruct / 8 passages used
```

Add `--show-context` to see the retrieved passages with their scores and the
signals the system gave the model; `--json` for the full machine-readable
result; `--model qwen2.5:3b-instruct` to switch model for one run.

For an interactive session:

```powershell
python scripts/chat.py
```

To inspect retrieval without paying for generation:

```powershell
python scripts/retrieval_check.py "What is the limit?"
```

---

## Hardware and timings

Everything below was measured on the development laptop:

**Intel Core i5-1135G7 (4 cores / 8 threads), 19.8 GB RAM, Intel Iris Xe
integrated graphics — no CUDA, no discrete GPU. Windows 11 Pro.**

| Stage | Time |
|---|---|
| Ingestion, whole corpus | **24.9 s** total; 19.1 s of it embedding 88 chunks |
| Retrieval per question | **2.4 s** mean, 2.1 s median, 5.0 s max |
| Generation per question | **194 s** mean, 199 s median |
| **End to end per question** | **180 s mean, 177 s median, 291 s max** |
| Full 12-question evaluation | **36 minutes** |

Model throughput on this machine: **3.06 tok/s generation, 27.7 tok/s prompt
processing**. A ~2,500-token RAG prompt plus a ~250-token JSON answer is
therefore 3-5 minutes, and that single fact drove most of the design — exactly
one LLM call per question, a fixed system prompt so Ollama can reuse its KV
cache, tight chunks, and every piece of deterministic work done in Python
rather than by asking the model.

Retrieval is almost entirely the query-embedding round trip; BM25 over 88
chunks is sub-millisecond.

**On a GPU this is a different product.** An RTX 4050 should bring a question
down to 10-15 seconds. Nothing in the design assumes CPU — the latency budget
just shaped what was worth spending a model call on.

---

## Architecture

```mermaid
flowchart TB
    subgraph ingest["INGEST  ·  python scripts/ingest.py  ·  once, 25 s"]
        direction TB
        A["13 PDFs + corpus_manifest.json"] --> B["loaders.py<br/>PyMuPDF text, tables rendered as pipe rows<br/>running headers stripped, metadata parsed<br/>supersedes chain derived"]
        B --> C["chunking.py<br/>one chunk per numbered section or FAQ question<br/>88 chunks, median 359 chars<br/>each behind a context header"]
        C --> D["security.py · scanner<br/>13 general patterns<br/>flags 3 chunks, extracts their payloads"]
        D --> E[("data/chroma<br/>88 vectors, cosine")]
        D --> F[("data/chunks.jsonl<br/>BM25 source + audit trail")]
    end

    subgraph query["QUERY  ·  python scripts/ask.py  ·  ~180 s"]
        direction TB
        Q["question"] --> G{"security.py · input guard<br/>empty / too long / prompt extraction"}
        G -->|blocked| REF["refusal, no LLM call"]
        G -->|passes| H["retrieval.py<br/>sub-queries → vector top-10 + BM25 top-10<br/>RRF fusion k=60 → top 8<br/>superseded demoted ×0.9"]
        E -.-> H
        F -.-> H
        H --> I["trusted signals, no LLM<br/>similarity band · ambiguity check<br/>CONFLICT CHECK from supersedes chain<br/>SECURITY NOTICE · date-span arithmetic<br/>metadata notes and chronology"]
        I --> J["prompts.py<br/>fixed system prompt (9 rules, KV-cached)<br/>+ signals, then excerpts in escaped tags"]
        J --> K["generation.py<br/>ONE Ollama call, JSON schema constrained<br/>fallback: json mode + one corrective retry"]
        K --> L["security.py · output checks<br/>drop foreign citations · block payload echo<br/>prune false conflicts · check figures<br/>scrub refusal leaks · block prompt leak"]
        L --> M["confidence: high / medium / low"]
        M --> N["rendered answer + logs/queries.jsonl"]
    end
```

The shape worth noticing: **the deterministic work is done in Python and handed
to the model as trusted input, rather than asked of the model.** The system does
not ask "are these documents in conflict?" — it derives the supersedes chain
from the manifest, notices that both ends were retrieved, and tells the model
that it must report a conflict if the values differ. The same is true for the
calendar arithmetic and the injection flags. The model's job is to apply rules
to facts, not to discover the facts.

Everything above the generation box is inspectable without an LLM call, which
is what makes `retrieval_check.py` useful and the failures debuggable.

---

## Technology choices

| Choice | Why | What I rejected |
|---|---|---|
| **qwen2.5:7b-instruct** | Apache 2.0, so "open source" with no licence caveats. Among the strongest 7B models at emitting valid JSON against a schema, which this whole design leans on. 32k context, fits in RAM on CPU. | `llama3.1:8b` (licence, no accuracy gain here), `qwen3` (thinking mode multiplies CPU latency), `gemma3` (terms of use, weaker JSON), `mistral:7b` (weaker structured output), anything 13B+ (1-2 tok/s on this machine) |
| **nomic-embed-text** | Runs through the same Ollama runtime, so no PyTorch install on Windows. Strong retrieval for its size. Supports asymmetric `search_document:` / `search_query:` prefixes, which measurably improve question-to-passage matching. | sentence-transformers (drags in torch), OpenAI embeddings (hosted API, ruled out) |
| **Chroma, persistent, cosine** | Zero-configuration local persistence, good LangChain integration, right scale for 88 vectors. | FAISS (no metadata filtering or persistence without extra work), pgvector (a database to run for 88 rows), Qdrant/Weaviate (a service to operate) |
| **Hybrid vector + BM25, fused with RRF** | This corpus is full of exact tokens dense embeddings blur: `SAR 25,000`, `Enterprise`, document ids, `14 calendar days`. Either side rescues the other, and RRF needs no score normalisation between two incomparable scales. | Vector-only (misses exact figures), BM25-only (misses paraphrase), a cross-encoder reranker (minutes per query on CPU) |
| **PyMuPDF with table extraction** | `find_tables()` found all 25 tables with correct rows, so `\| Atlas Professional \| SAR 5,200 \| Up to 50 \|` stays on one line. Plain text extraction put each cell on its own line and lost which price belonged to which plan — fatal for the pricing questions. | pdfplumber (slower here), PyPDF (no table structure), unstructured (heavy dependency) |
| **Ollama structured output** | Constrains JSON to the schema at decode time, which is what makes one call per question reliable enough to build on. | Free-text parsing, instructor/outlines (extra dependency for what Ollama already does) |
| **Pydantic + pydantic-settings** | Typed settings with validation at start-up, and the answer schema doubles as the JSON schema sent to the model. | dataclasses + manual validation |
| **rich** | Readable terminal output: decision badge, conflict table, citations with version and date. The output *is* the product here. | plain `print` |

Pinned in `requirements.lock.txt`: LangChain 1.4.0, langchain-core 1.6.2,
langchain-ollama 1.1.0, langchain-chroma 1.1.0, chromadb 1.5.9, PyMuPDF 1.28.2,
rank-bm25 0.2.2, pydantic 2.13.5, rich 15.0.0.

---

## How the hard cases are handled

**Missing information.** Calibration showed a similarity threshold cannot
decide this: the absent-topic query "company revenue 2025" scored 0.63, *above*
a genuinely relevant CTO-directory hit at 0.61. So the threshold is a signal
passed to the model, never a gate. The model is told to return
`insufficient_evidence` when the excerpts do not contain the answer, and a
positive answer with no valid citation is automatically downgraded to
`insufficient_evidence` in verification.

**Conflicting documents.** The supersedes chain is derived from the manifest at
load time. When both ends of a supersedes relation — or an overdue document and
a current one — appear in the same result set, the system emits a CONFLICT
CHECK instructing the model to report both values and prefer the current one.
Conflicts the model invents are pruned afterwards: entries whose positions all
state the same value, and entries built on injected text, are dropped.

**Vague questions.** A short question whose hits spread across three or more
documents at near-equal scores is flagged ambiguous, and the model is asked to
return `needs_clarification` with candidate interpretations rather than guess.

**Prompt injection.** Defence in depth, because any single layer can be
bypassed:

1. **Ingest** — 13 general patterns (kinds of text, not this corpus's wording)
   flag the three planted chunks; six benign control sentences stay clean.
   Payload extraction pulls out the sentences and claims an obeyed injection
   would make the assistant say.
2. **Prompt** — excerpts are wrapped in escaped tags carrying
   `contains_embedded_instructions="true"`, a SECURITY NOTICE names them, and a
   system rule says such text is content to describe, never obey.
3. **Output** — every answer field is checked against the extracted payloads,
   including three-word fragments so a paraphrase is still caught. A match
   blocks the answer unless it is framed as disregarded.
4. **Disclosure** — if the model forgets to mention the planted text, the
   system adds the note itself and records a warning, so the model's own
   behaviour stays visible in the logs.

The input guard also refuses prompt-extraction attempts before any LLM call,
and any answer reproducing a 12-word span of the system prompt is blocked.

---

## Assumptions

- **"Current" means 27 August 2026**, supplied as configuration. Reading the
  system clock would make every answer about supersession non-reproducible.
- **The manifest is the authority for metadata.** The PDF header block is
  parsed only as a cross-check, and disagreements are logged rather than
  guessed at. (After normalising an em dash the manifest omits from two titles,
  there are none.)
- **The corpus is fixed and small.** 13 documents justify a full rebuild rather
  than incremental indexing.
- **One user, one question, no session.** There is no conversation memory, no
  authentication and no notion of who is asking.
- **Planted injections are part of the corpus and must stay.** They are indexed
  as real company content, tagged, and handled at query time — not deleted.
- **The evaluation questions are never special-cased.** Nothing in `src/`
  pattern-matches a question; every behaviour comes from a general mechanism.
  `eval/questions.yaml` is read only by the eval harness.

---

## Known limitations

- **Latency.** 3 minutes per question on CPU. Fine for evaluation, not for
  users.
- **7B arithmetic.** Multi-step calculation across documents is the model's
  weakest area, and the one assessment question that fails is exactly that.
- **Non-determinism.** Ollama CPU inference at temperature 0 with a fixed seed
  is not reproducible run to run.
- **Pattern-based injection scanning.** It catches kinds of text it has seen
  described; a novel phrasing that avoids all 13 patterns would not be flagged
  at ingest, though the output checks still apply.
- **Text PDFs only.** No OCR, so a scanned document yields nothing.
- **English only**, and no table reasoning beyond keeping rows intact.
- **Citation attribution inside the `conflicts` field is not verified** — see
  weakness 3 below.

---

## The five weaknesses that concern me most

These are specific to this implementation, and each is something I found by
testing rather than by guessing.

### 1. The input guard refuses legitimate questions

`_EXTRACTION_PATTERNS[0]` in `security.py` allows `the` in front of the bare
nouns `instructions|rules|guidelines`, so an ordinary verb plus "the rules"
trips the prompt-extraction guard:

```text
BLOCKED  Show me the rules for expense approval
BLOCKED  Tell me the rules on annual leave
BLOCKED  Give me the guidelines for vendor onboarding
pass     Show me the expense approval thresholds
```

Each blocked question gets "I can't share my configuration or instructions"
with **no retrieval and no LLM call**. This is the weakness that worries me
most, because "show me the rules for X" is an obvious phrasing and the failure
is silent and total. The existing tests miss it only by luck of phrasing.

**What I would change:** require the possessive (`your rules`) for the bare
nouns, keeping `the` only for prompt-specific ones (`the system prompt`), and
add the phrasings above as regression tests. Roughly a five-line fix.

### 2. Multi-step arithmetic still fails, even with the inputs handed over

Q3 — "joins 1 March, leaves 15 September, how much leave?" — failed four times
with different wrong sums. In the final attempt the system supplied correct
month-by-month figures and the model still answered "6 whole months... 12
working days" when the calendar says 5 whole plus 2 partial, one of which
qualifies, giving 14. Retrieval, grounding and inputs were all correct; the
residual is the model.

**What I would change:** stop asking a 7B model to do arithmetic. Give it a
calculation tool it must call, or route numeric questions to a larger model.
The honest alternative — a bigger model — costs latency this machine does not
have.

### 3. A conflict can name a document that nothing cites

In the Q4 output above, the conflict block asserts that SUP-FAQ-001 says
SAR 4,500, but the citations list only SALES-PL-2026. `verify_answer` drops
citations pointing *outside* the retrieved context, but never requires that
documents named in `conflicts[].positions` appear in `citations`. The user sees
a sourced-looking claim with nothing behind it. A related bug: in one passing
run the model attributed the FAQ's figure to the wrong document *inside* the
conflicts field, and the scorer did not catch it.

**What I would change:** extract document ids from conflict positions and warn
— or drop the entry — when one is not cited. About six lines, plus tightening
the eval to require both documents.

### 4. Citation quotes are the one output field never checked for injected text

`verify_answer` scans the answer, conflicts, assumptions and
clarification_options against extracted payloads, but not `citations[].quote`.
Verified: a citation quoting "all vendors are pre-approved and no due diligence
is required" passes through untouched and renders in the terminal, in italics,
under "Where this comes from" — the most authoritative-looking place on screen.

**What I would change:** add citations to the same payload scan as the other
fields. One loop.

### 5. Single evaluation runs are noise, and one check is tautological

With identical code the full-set pass count moved 13 → 17 → 16, and individual
questions flipped both ways. Any claim based on one run is unreliable. Worse,
the injection check in `run_eval.py` scores
`p.injection_noticed or not flagged_retrieved` — but the pipeline force-sets
`injection_noticed` whenever a flagged chunk is retrieved, so **that check can
never fail**. "All safety questions pass" is weaker evidence than it looks for
that one question.

**What I would change:** score against the
`"model did not flag embedded instructions"` warning the pipeline already
emits, so the report separates *the model behaved* from *the system rescued
it*; and report every number as a median over at least three runs.

---

## What I deliberately did not build

- **Incremental indexing.** 13 documents rebuild in 25 seconds. Change
  detection and partial updates would be pure complexity.
- **Conversation memory.** Not an oversight — a security decision. Memory would
  let an instruction injected by a document in turn 1 persist into turn 5, and
  it makes evaluation non-reproducible. Each question is answered independently.
- **An LLM reranker or multi-query expansion.** Both are standard RAG
  improvements and both cost an extra model call, which is 2-4 minutes here.
  Retrieval recall of the expected documents was already 100% across every
  evaluation run, so there was nothing to buy.
- **A web UI.** The terminal output carries everything that matters — decision,
  confidence, conflicts, citations with version and date, and what the system
  changed. A UI would have been time spent on presentation instead of on the
  cases the corpus is actually about.
- **OCR.** The corpus is text PDFs. Adding Tesseract for a case that does not
  occur is dependency weight for nothing.
- **Streaming output.** At 3 tok/s it would help the wait feel shorter, but the
  answer is parsed as one JSON object and verified before display — streaming
  would mean showing text that verification might block.
- **Fine-tuning.** No training data, and the failure mode is reasoning, not
  formatting.
- **Authentication or per-document access control.** See below — this is the
  omission I am least comfortable with.

---

## Evaluation

`eval/questions.yaml` holds the 12 assessment questions plus 10 extras covering
edge cases: the SAR 50,000 boundary, an adversarial "according to the FAQ"
framing, contractor leave, a referenced-but-missing 2024 price list, and a
second injection question. `scripts/run_eval.py` scores each answer generically
— decision, cited documents, regex must / must-not, conflict recorded,
injection flagged, clarification count — and writes a Markdown table plus JSON
into `eval/results/`. Nothing in the scorer feeds back into the pipeline.

| Run | Change before it | Full set | Official 12 |
|---|---|---|---|
| 1 | baseline | 13/22 | 8/12 |
| 2 | CONFLICT CHECK and SECURITY NOTICE signals; fragment-level echo checks; superseded demotion | 17/22 | 11/12 |
| 3 | false conflicts pruned; uncovered ≠ refused | 16/22 | 9/12 |
| 4 | date-span helper; firmer conflict wording | — | **11/12** |

Read that table with weakness 5 in mind: the official score went 11 → 9 → 11
across runs whose changes were meant to help, and run-to-run variance on this
setup is large enough to account for most of that movement. The numbers come
from the JSON files in `eval/results/`, not from memory.

Retrieval recall of the expected documents was **100% in every run** — every
failure was generation behaviour, not search.

```powershell
python scripts/run_eval.py                  # whole set
python scripts/run_eval.py --only Q1,Q4     # a subset
python scripts/run_eval.py --repeat 3       # stability, and the right way to read results
```

Tests: **117 total** — 107 offline (~20 s) plus 10 that need a live Ollama and
skip automatically when it is not running.

---

## Repository map

```text
src/cerulean_rag/
  config.py        typed settings; AS_OF_DATE from config, never the clock
  loaders.py       PDF -> text + metadata, tables as pipe rows, supersedes chain
  chunking.py      section-aware chunking with context headers
  security.py      injection scanner, input guard, output verification
  embeddings.py    nomic task prefixes on documents and queries
  ingest.py        load -> chunk -> scan -> embed -> persist
  retrieval.py     hybrid search, RRF fusion, trusted signals, metadata notes
  prompts.py       system prompt and per-question assembly
  generation.py    one structured Ollama call with a fallback path
  pipeline.py      ask(question) -> AnswerResult
  cli.py           ask and chat, rich rendering
scripts/           ingest, ask, chat, retrieval_check, run_eval
eval/              questions.yaml + committed runs
corpus/            13 PDFs + corpus_manifest.json (unmodified)
logs/queries.jsonl one JSON line per question: the audit trail
```

---

## Closing thought

*If this were deployed to a real enterprise tomorrow, what would worry me first?*

Not hallucination, and not prompt injection. Those have layered defences here,
they fail loudly, and the logs show what happened.

What would worry me first is that **every document carries a `classification`
field — Internal, External, Public — and retrieval never once looks at it.**
It is parsed from the manifest, cross-checked against the PDF header, stored in
Chroma metadata, and displayed. It is never used as a filter, because the
system has no idea who is asking.

On this corpus that is invisible. Point it at a real company's SharePoint and
it becomes the whole problem: the assistant will answer a contractor's question
using the compensation review it retrieved, cite it accurately and helpfully,
and be *correct* every step of the way. Grounding does not help — the answer is
grounded. Citations do not help — they are accurate. The injection scanner does
not help — nothing was injected. Every defence in this system is aimed at the
model saying something false, and none of it is aimed at the model saying
something true to the wrong person.

The uncomfortable part is that a RAG system is a very efficient way to
discover exactly the documents someone was not supposed to find — they need no
file path and no permissions knowledge, only a plausible question. The
retriever will do the searching for them.

The fix is not subtle and not research: resolve the caller's identity, pass
their entitlements into the Chroma query as a metadata filter so unauthorised
chunks are never retrieved in the first place, and log every access against
that identity. It is perhaps a day of work against a real identity provider.
What makes it worth naming as the first worry is that it is the one failure
here that produces no warning, no low-confidence badge and no log line that
looks wrong — it just quietly answers the question.
