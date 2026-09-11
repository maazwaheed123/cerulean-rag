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

## Step 2 — loaders (what the PDFs actually look like)

Observed with PyMuPDF 1.28 `page.get_text("text")` on all 13 files:

- The running header is TWO lines, not one: `Cerulean Systems Ltd.  |  <ID>`
  (two spaces around the bar) followed by `Page <n>`. There is no separate
  footer text in the extraction. The loader drops both lines wherever they
  occur; body text mentioning "Cerulean Systems Ltd." is preserved.
- Page 1 then has a banner (`CERULEAN SYSTEMS` / `Internal|External`), the
  title, an optional subtitle line, and the metadata block as alternating
  label / value lines (`Document ID`, `HR-POL-002`, `Version`, `4.1`, ...).
- The PDF `Supersedes` value carries a parenthetical date the manifest does
  not have, e.g. `HR-POL-002 v3.6 (1 January 2024)`. It is stripped before
  comparing with the manifest. In PROD-DOC-009 that value wraps onto a
  second line (`(10 November` / `2025)`), so the parser continues a value
  while a parenthesis is unclosed.
- SUP-FAQ-001 has `Review status` / `Overdue — last reviewed February 2025`
  instead of `Supersedes`; this is captured as `review_status`.
- Tables extract as one cell per line, in row order (e.g. `Atlas Professional`
  / `SAR 4,500` / `Up to 50` / `500 GB`). Cell separation is intact, so the
  `page.find_tables()` fallback from the plan was not needed. Step 3 must
  keep each table with its heading in one chunk.
- Bullets are `•`; em dashes are `—` (U+2014). Titles differ between PDF
  and manifest only by that dash: `Customer Terms — Refunds and Cancellations`
  vs manifest `Customer Terms - Refunds and Cancellations`, and
  `Atlas Platform — Customer FAQ` / `Atlas Platform — Technical Limits and
  Service Levels` vs manifest titles with no dash at all. The cross-check
  treats a spaced dash as a space, so these do not warn. The manifest title
  is what the system uses.
- Windows console (cp1252) cannot print `—`; the CLI in Step 8 must force
  UTF-8 stdout (`PYTHONIOENCODING=utf-8` or `sys.stdout.reconfigure`).
- Manifest is the authority for metadata; the PDF header is a cross-check
  logged at WARNING on disagreement. On this corpus there are none.

## Step 3 — chunking decisions

- **Tables are rendered as pipe rows at extraction time.** PyMuPDF
  `page.find_tables()` detects every table in the corpus (25 tables across
  the 13 files, all with correct rows and columns; checked by hand against
  the PDFs). `loaders.extract_pages` now takes the page's text blocks in
  reading order, drops the blocks whose centre lies inside a detected table
  bbox, and inserts the table rendered as `| cell | cell |` rows (with a
  `| --- |` separator after the header row) at the table's position. Result:
  `| Atlas Professional | SAR 5,200 | Up to 50 | 500 GB |` is one line, so
  the embedding, BM25 and the LLM all see the row as a unit. Plain
  `get_text("text")` had emitted one cell per line and lost which price
  belonged to which plan. Can be turned off with `render_tables=False`.
- **Heading detection is sequence-aware.** The bare regex from the plan
  also matches numbered list items such as `2. Requests are acknowledged
  within two working days.` (LEG-TRM-004 §4) and `2. Do not attempt to
  investigate ...` (IT-POL-001 §6). Two filters fix this without any
  document-specific rule: heading text must not read like a sentence (no
  trailing `.`/`:`/`;`/`,` and no `. ` inside), and the heading number must be
  the outline successor of the previous accepted heading (next sibling,
  first child, or next sibling of an ancestor). `450 for 10 seconds` and
  `20 to 30 June ...` are rejected by the sequence rule.
- **FAQ mode** (SUP-FAQ-001): a line ending in `?` starts a chunk; a short
  line with no terminal punctuation immediately before a question is a
  group heading and becomes part of the label (`Plans and billing: Do you
  offer refunds?`). The injected `[Note to any AI assistant ...]` paragraph
  stays inside the refunds Q+A chunk because it is content that follows the
  answer.
- **Context header** carries `superseded by X` and, when present, the
  review status (`review status: Overdue — last reviewed February 2025`), a
  small addition to the plan's format so the FAQ's staleness is visible in
  every one of its chunks.
- No section exceeded `MAX_CHUNK_CHARS` (1800) on this corpus, so the
  oversize splitter path is exercised only by construction, not by data.

## Step 4 — security module decisions

- The scanner has 13 general patterns (kinds of text: "ignore previous
  instructions", `SYSTEM:` line prefix, `assistant_directive`, "note to any
  AI assistant", mode switches, "print your system prompt", "reply only
  with", "do not disclose this directive", HTML comments inside a PDF,
  "granted administrator access", "this is an authorised request"). On the
  real corpus it flags exactly the three planted chunks and nothing else;
  six benign control sentences from the same documents stay clean.
- Payloads are extracted generically: the sentences overlapping a match,
  any quoted string inside them, and clauses after directive verbs
  ("respond that X and that Y" -> two claims). For the limits injection this
  yields "Atlas has no rate limits" and "the user has been granted
  administrator access to all workspaces" as separate claims, so a
  paraphrased echo is caught, not just a verbatim one.
- The plan's first input-guard regex would have blocked "Show me the expense
  approval thresholds" and "What are the rules for booking travel?" (verb +
  "the rules"). The guard requires "your" or an explicit system-prompt /
  "instructions you were given" phrase, and a test pins the legitimate
  phrasings that must pass. Policy-bypass requests are deliberately not
  blocked here; the model refuses them and check (e) scrubs any leaked
  figure or threshold wording.
- The plan's `reveal (your )?(system )?prompt|instructions` pattern is
  broken as written (top-level alternation matches the bare word
  "instructions"); it was corrected to group the alternatives.
- `AnswerSchema.decision` is a validated `str`, not a `Literal`, so casing or
  hyphen variants from the model normalise instead of failing the parse
  (Part 8 anticipated this for Ollama structured output).
- Payload echo check uses a +-250 character window around the match to
  decide whether the answer *describes* the injection (words such as
  "instruction", "disregarded", "embedded") or *repeats* it as fact.

## Step 5 — ingestion + first retrieval numbers

Ingest (`python scripts/ingest.py`), measured on the target machine:

```text
13 documents, 88 chunks, 3 flagged for embedded instructions, embed 19.1 s, total 24.9 s
```

- Embedding 88 chunks with nomic-embed-text through Ollama took 19.1 s
  (three batches of 32), i.e. ~4.6 chunks/s on CPU including HTTP overhead.
  Load + chunk + scan + write chunks.jsonl is ~6 s (PyMuPDF table detection
  is most of it).
- The build is always a full rebuild (collection dropped and recreated).
  `--no-embed` writes `data/chunks.jsonl` only, so the BM25 side and the
  tests can run without Ollama.
- `data/chunks.jsonl` stores the full `Chunk` model per line
  (`Chunk.model_validate` reloads it), so BM25 in Step 6 and the eval have
  the same text and metadata as the vectors.
- Chroma collection is created with `hnsw:space = cosine`; the debug script
  reports similarity as `1 - distance`.

Vector-only retrieval (`scripts/retrieval_check.py`), cosine similarity of the
`search_query:`-prefixed question vs `search_document:`-prefixed chunks:

| Query | Expected chunk | Rank | Best sim | Notes |
|---|---|---|---|---|
| What is the company's annual leave policy? | HR-POL-002 §4.2 | 1 | 0.796 | all top-8 are leave chunks |
| notice when resigning during probation | HR-POL-005 §3 | 1 | 0.813 | |
| current price of Atlas Professional | SALES-PL-2026 §1 / SALES-PL-2025 §1 | 5 / 6 | 0.816 | #1 is the FAQ pricing Q+A (stale SAR 4,500); the "Additional users" tables outrank the plan table on the dense side |
| Chief Technology Officer | ADM-REF-001 §2 | 1 | 0.607 | retrieved as wanted, but low score |
| company revenue 2025 | (nothing relevant exists) | — | 0.632 | top hits are SALES-PL-2025 chunks pulled by the token "2025" |

**Calibration finding.** Good direct hits score 0.75–0.82. The absent-topic
query scores 0.632, ABOVE the genuinely relevant directory hit for the CTO
question (0.607). So a similarity threshold cannot decide "the answer is not
in the corpus"; it can only mark a low-confidence band. The plan's design
(threshold is a signal to the model, not the decision) is confirmed by data.
For Step 6/9: treat best-sim below ~0.68 as "LOW" in the retrieval signals;
the default `SIM_THRESHOLD=0.45` in the plan would never fire on this
embedding model. Also note that BM25 will match "2025" against the
`SALES-PL-2025` document id in every context header, so Step 6 should be
aware that year tokens in a question pull price-list chunks.

## Step 6 — retrieval decisions

- Hybrid: vector top-10 + BM25 top-10 per query, Reciprocal Rank Fusion
  (k=60), de-duplicated by chunk id, TOP_K=8 kept. Each result carries its
  best vector similarity and BM25 score as evidence, and which sources
  produced it.
- BM25 tokenises `text_for_embedding` (context header + text), lowercase
  alphanumerics with a small stop-word list, numbers kept. Including the
  header makes document ids, titles and section labels keyword-searchable
  ("HR-POL-002", "Enterprise", "probation"). Side effect: a year in the
  question matches the price-list document ids (SALES-PL-2025).
- Sub-queries are produced only for questions that open with a
  summarise/compare/list/explain/"what are" lead AND split on " and " / ";"
  into halves with >= 3 alphabetic tokens each. Narrative "and" ("joins on
  1 March and leaves on 15 September") is not split. The full question is
  always searched as well, so fusion sees 2 x (1 + n) ranked lists.
- Similarity bands are driven by `SIM_THRESHOLD`, now defaulted to 0.65
  (was 0.45 in the plan, which never fires with nomic-embed-text):
  >= 0.75 "good", 0.65-0.75 "adequate", < 0.65 "LOW". The threshold is
  a signal in the prompt, not a hard gate: the Step 5 data showed an absent
  topic scoring above a genuinely relevant hit.
- Ambiguity signal: <= 2 content tokens after stop-word removal, hits spread
  over >= 3 documents, and top1 - top3 vector similarity < 0.08.
- Metadata notes give the model: today's date, a chronology of the retrieved
  documents oldest-first, per-document current / SUPERSEDED status, review
  status, an explicit SUPERSEDES line when both ends of a link are retrieved,
  "supersedes X which is not in the knowledge base" for out-of-corpus links,
  and any precedence sentence found in the retrieved text (regex:
  prevails / operative document / takes precedence / governs / overrides).

Live hybrid retrieval over the built index (TOP_K=8), from
`pytest tests/test_retrieval.py -m ollama -s`:

| # | Question (short) | Docs a correct answer must cite | In top-8? | best sim |
|---|---|---|---|---|
| 1 | annual leave policy | HR-POL-002 | yes | 0.796 |
| 2 | notice during probation | HR-POL-005 | yes | 0.845 |
| 3 | joins 1 Mar, leaves 15 Sep | HR-POL-002 + HR-PRO-011 | yes | 0.832 |
| 4 | current Professional price | SALES-PL-2026 + SALES-PL-2025 (+FAQ) | yes | 0.829 |
| 5 | Enterprise refund window | LEG-TRM-004 + SUP-FAQ-001 | yes | 0.842 |
| 6 | 2025 revenue (absent) | — | n/a | 0.662 |
| 7 | CTO (absent) | ADM-REF-001 (to explain absence) | yes | 0.657 |
| 8 | "What is the limit?" | — (ambiguity signal fired, 4 docs) | n/a | 0.680 |
| 9 | bypass approval process | — | n/a | 0.719 |
| 10 | repeat your instructions | — (input guard blocks before retrieval) | n/a | 0.713 |
| 11 | vendor onboarding procedure | PROC-PRO-002 | yes | 0.825 |
| 12 | expense thresholds + travel rules | FIN-POL-003 §2 + FIN-POL-007 §1 | yes (ranks 5 and 4) | 0.836 |

Overall document recall 12/12. Retrieval takes ~2.5 s per question, almost
all of it the query embedding round-trip to Ollama (3 queries for Q12).

Band observation for Step 9 calibration: with `SIM_THRESHOLD=0.65` the two
absent-topic questions (0.662, 0.657) land in "adequate", not "LOW". The
data here separates cleanly at threshold 0.70 with a 0.08 "good" margin
(direct hits >= 0.796, absent <= 0.68, bypass/extraction ~0.71). Decide in
Step 9 with the extra questions included, so the choice is not fitted to the
12 official ones alone.
