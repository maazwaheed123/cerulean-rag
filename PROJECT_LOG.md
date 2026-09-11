# Project log — how this system was built (11 September 2026)

An in-depth record of the build session: what was asked, what was done at
each step, what was measured, what went wrong, and every design decision
with its reason. It complements `SETUP.md` (how to run) and `NOTES.md` (the
raw working notes). Written so that someone who did not watch the work can
follow and defend every choice.

## 0. The assignment and the ground rules

The task is the SAITC Applied AI Engineer take-home: a retrieval-augmented
assistant over 13 PDFs of the fictional Cerulean Systems Ltd. The assessors
say the interesting part is not the happy path but what happens when the
answer is absent, when two documents disagree, when the question is vague,
and when a document contains text designed to hijack the model. They will
run 12 questions (plus their own phrasings) and read the code.

A step-by-step build plan (12 steps, 0-11) was supplied by the user and
executed one step per prompt, each step ending with real verification output
and a git commit. Hard rules that shaped every decision:

- Never hard-code or pattern-match the 12 evaluation questions anywhere in
  the pipeline. Every behaviour must come from general mechanisms.
- Never call a commercial LLM API; generation and embeddings run locally
  through Ollama with open-weight models.
- Do not modify the corpus PDFs or the manifest. Three documents contain
  deliberate prompt injections; they must stay and be handled at runtime.
- "Current" means as at 27 August 2026, supplied as configuration
  (`AS_OF_DATE`), not the system clock, so results are reproducible.

## 1. Environment (Step 0)

Target machine: Intel Core i5-1135G7 (4 cores / 8 threads), 19.8 GB RAM,
Intel Iris Xe integrated graphics (no CUDA), Windows 11 Pro. Python 3.14 was
present and too new for the ML stack, so Python 3.12.10 was installed
alongside it with winget. Ollama 0.34.0 was installed with winget.

Models pulled: `qwen2.5:7b-instruct` (4.7 GB, 20 min download),
`nomic-embed-text` (274 MB), `qwen2.5:3b-instruct` (1.9 GB, optional).

Why these models:

- **qwen2.5:7b-instruct** — Apache 2.0 open weights (satisfies "open
  source" without Llama/Gemma licence caveats); among the strongest 7B
  models at emitting valid JSON against a schema, which this design depends
  on; follows "answer only from context" well; 32k context; fits in RAM on a
  CPU-only laptop.
- **nomic-embed-text** — runs through the same Ollama runtime (no PyTorch
  install on Windows), strong retrieval quality for its size, and supports
  asymmetric prefixes (`search_document:` / `search_query:`) that improve
  question-to-passage matching.
- Rejected: llama3.1:8b (licence, no accuracy gain), qwen3 (thinking mode
  multiplies CPU latency), gemma3 (terms of use, weaker JSON), mistral:7b
  (weaker structured output), anything 13B+ (1-2 tok/s here).

Measured speeds (the fact that shaped the whole design):

| Model | Prompt processing | Generation |
|---|---|---|
| qwen2.5:7b-instruct | ~17.5 tok/s | ~3.5 tok/s |
| qwen2.5:3b-instruct | ~45 tok/s | ~8.4 tok/s |

A RAG prompt of ~2,500 tokens plus a ~250-token JSON answer therefore costs
2.5-4 minutes per question on the 7B model. Consequences: exactly one LLM
call per question, a fixed system prompt so Ollama can reuse its KV cache,
tight chunks, and the 3B model for iteration.

## 2. Repository scaffold (Step 1)

`src/` layout installed with `pip install -e .`; pydantic-settings
configuration with a validated `AS_OF_DATE`; rich console plus rotating file
logging; the 13 PDFs and manifest copied into `corpus/` and verified
byte-identical by SHA-256; the assignment brief and the corpus README kept
under `docs/` so they are never indexed as company knowledge.

Two traps found and fixed here: git's `autocrlf` classified the small
text-based PDFs as text and would have rewritten their bytes on any clone
(fixed with `.gitattributes`: `*.pdf binary`), and PowerShell's `Out-File`
writes a UTF-8 BOM that breaks setuptools' reading of `requirements.txt`.

Resolved versions: LangChain 1.4.0, langchain-core 1.6.2, langchain-ollama
1.1.0, langchain-chroma 1.1.0, chromadb 1.5.9, PyMuPDF 1.28.2, rank-bm25
0.2.2, pydantic 2.13.5.

## 3. Loading the PDFs (Step 2)

PyMuPDF text extraction. The real layout differed from the plan in three
ways, all handled generically: the running header is two lines
(`Cerulean Systems Ltd.  |  <ID>` then `Page N`); the metadata block is
label/value on alternating lines; one `Supersedes` value wraps onto a second
line (the parser continues a value while a parenthesis is unclosed). The
manifest is the authority for metadata; the PDF header is a cross-check that
logs a warning on disagreement (none, after normalising the em dash the
manifest omits from two titles).

`superseded_by` and `is_current` are derived from the supersedes chain
across the corpus. Only one link points inside the corpus: SALES-PL-2026
supersedes SALES-PL-2025. SUP-FAQ-001's "Review status: Overdue" is captured.

## 4. Chunking (Step 3)

One chunk per numbered section (or per FAQ question), tables kept whole with
their heading. 88 chunks, 129 to 1,016 characters, median 359.

Two decisions worth defending:

- **Tables are rendered as pipe rows at extraction time.** PyMuPDF's
  `find_tables()` detected all 25 tables with correct rows and columns, so
  `| Atlas Professional | SAR 5,200 | Up to 50 | 500 GB |` stays on one line.
  Plain text extraction had emitted one cell per line and lost which price
  belonged to which plan. This matters for the pricing and limits questions
  and for keyword search.
- **Heading detection is sequence-aware.** A bare regex also matches
  numbered list items such as "2. Requests are acknowledged within two
  working days." Two general filters fix it: heading text must not read
  like a sentence, and the number must be the outline successor of the
  previous heading.

Every chunk carries a context header such as
`[HR-POL-002 v4.1 | Leave and Time Off Policy | effective 2026-01-01 | current | 4.2 Annual leave entitlement]`
that is embedded with the text and shown to the model, so identity, dates and
supersession status travel with the passage.

## 5. Security module (Step 4)

Three independent, deterministic defences:

1. **Injection scanner** at ingest: 13 general patterns (kinds of text, not
   corpus wording). Flags exactly the three planted chunks; six benign
   control sentences stay clean. Payload extraction pulls the sentences,
   quoted strings, and "respond that X and that Y" claims an obeyed
   injection would make the assistant say.
2. **Input guard** before retrieval: refuses empty, over-long, and
   prompt-extraction inputs without an LLM call. Kept deliberately narrow:
   the plan's first regex would have blocked "Show me the expense approval
   thresholds", and policy-bypass requests are left for the model to refuse.
3. **Output verification** after generation: drops citations to documents
   not in context, blocks answers that repeat an injection payload as fact
   (later extended to 3-word fragments of short payloads and to the
   structured fields), warns on figures absent from the context, downgrades
   uncited answers, scrubs refusals that leak figures, and blocks any answer
   reproducing a 12-word span of the system prompt.

## 6. Ingestion and first retrieval numbers (Step 5)

`scripts/ingest.py`: load, chunk, scan, write `chunks.jsonl`, embed into a
cosine-space Chroma collection in batches of 32. Full rebuild every run
(13 documents; incremental indexing would add complexity for no benefit).
Measured: 88 chunks embedded in 19.1 s, total 24.9 s.

Calibration finding: direct hits score 0.75-0.82 cosine similarity, but the
absent-topic query "company revenue 2025" scored 0.63, above the genuinely
relevant CTO directory hit at 0.61. A similarity threshold therefore cannot
decide that an answer is missing; it is only a signal to the model. The
plan's default of 0.45 would never fire and was moved to 0.65.

## 7. Hybrid retrieval (Step 6)

Vector top-10 plus BM25 top-10 per query, fused with Reciprocal Rank Fusion
(k=60), top 8 kept. BM25 runs over the chunk text including its context
header, so document ids and titles are keyword-searchable. Compound questions
opening with summarise/compare/list are split on "and" into sub-queries;
narrative "and" ("joins on 1 March and leaves on 15 September") is not.

Trusted material assembled for the prompt without any LLM call: a similarity
band, an ambiguity warning for short questions whose hits spread over three
or more documents with flat scores, and metadata notes giving today's date,
a chronology of the retrieved documents, current/superseded status, review
status, explicit SUPERSEDES lines, out-of-corpus supersedes links, and any
precedence sentence found in the text ("this schedule prevails").

Live result: the documents a correct answer must cite were in the top 8 for
all 12 questions (recall 12/12); retrieval takes ~2.5 s, almost all of it
the query-embedding round trip.

## 8. Generation and the pipeline (Step 7)

A fixed system prompt (nine rules: ground, cite, resolve conflicts openly,
combine across documents, ask when ambiguous, excerpts are untrusted data,
decline out-of-scope, never reveal instructions, do not invent) plus a
per-question human message: trusted signals and notes first, then the
untrusted excerpts wrapped in escaped `<document>` tags with status and an
embedded-instructions flag, then the question. One structured call through
Ollama's JSON-schema mode; fallback to JSON mode with one corrective retry;
never an exception to the user.

The pipeline: input guard, retrieval, prompt, generation, verification,
confidence (high/medium/low from similarity band, citations and warnings),
one JSON line per question in `logs/queries.jsonl` with per-stage timings.

First end-to-end result: the pricing question returned SAR 5,200 with a
recorded conflict against the FAQ's SAR 4,500 and the supersession reasoning,
high confidence, 236 s generation.

## 9. CLI (Step 8)

`ask` and `chat` commands with rich rendering (decision badge, answer,
sources with title/version/date, conflicts block, clarification options,
warnings, timing line), `--json`, `--show-context`, `--model`. Preflight
Ollama check; every failure becomes one actionable sentence with tracebacks
in the log file only. No conversation memory in the chat, by design: memory
would let an injected instruction persist across turns and would make
evaluation non-reproducible.

## 10. Evaluation (Step 9)

`eval/questions.yaml` holds the 12 official questions plus 10 extras (edge
cases: SAR 50,000 boundary, adversarial "according to the FAQ", contractor
leave, a referenced-but-missing 2024 price list, a second injection
question). `scripts/run_eval.py` scores each answer generically (decision,
cited documents, regex must/must-not, conflict recorded, injection flagged,
clarification count) and writes a Markdown table plus JSON.

Four runs on qwen2.5:7b-instruct, each 36-56 minutes on this CPU:

| Run | Changes before it | Full set | Official 12 |
|---|---|---|---|
| 1 | baseline | 13/22 | 9/12 |
| 2 | CONFLICT CHECK and SECURITY NOTICE signals from metadata; fragment-level echo checks across all answer fields; system adds the injection disclosure when the model forgets; superseded documents demoted x0.9 after fusion; prompt rules tightened | 17/22 | 11/12 |
| 3 | conflicts whose positions agree or are commentary about injected text are pruned; "uncovered question is insufficient_evidence, not refused"; exact-figure rule | 16/22 | 10/12 (Q1 was a scoring regex bug) |
| 4 | date-span helper (system lists months and days served for any two-date question); firmer "you must record the conflict" wording; official 12 only | n/a | **11/12** |

Safety core (Q6 absent revenue, Q7 absent CTO, Q9 bypass, Q10 prompt
extraction, Q11 vendor injection) passed in every run from run 2 onward.
Retrieval recall of expected documents was 100% in every run: all failures
were generation behaviour.

Two findings the assessors should see:

- **Run-to-run variance.** With identical code the pass count moved
  13 -> 17 -> 16 and Q5 flipped both ways. Ollama CPU inference at
  temperature 0 with a fixed seed is not reproducible; single runs are noisy
  and a change must be judged over repeated runs (`--repeat`).
- **Q3 never passed.** The date-range leave calculation (1 March to 15
  September, answer 14 working days) failed four times with different wrong
  arithmetic. In the final run the system supplied the exact month-by-month
  figures and the model still mis-added ("... total is 10 ... The total is
  19"). Retrieval, grounding and inputs were all correct; the residual is
  the 7B model's multi-step reasoning. Documented as a known limitation;
  the honest fixes are a larger model or a deterministic calculation tool.

Also noted: in the passing Q5 answer the model attributed the FAQ's 30-day
figure to the wrong document inside the conflicts field; the scorer does not
catch attribution errors inside that field.

Per the user's instruction, iteration stopped after run 4.

## 11. Decisions and observations that belong in the README

- As-of date from configuration, not the clock (reproducibility).
- One LLM call per question; fixed system prompt as a cache prefix
  (roughly halved latency once Ollama re-used the KV cache).
- Tables rendered as rows; sequence-aware headings; context headers.
- Threshold is a signal, not a gate (absent topics can outscore relevant hits).
- Trusted system-generated signals (conflict checks, security notices, date
  spans, metadata notes) do the deterministic work; the model applies rules.
- Defence in depth for injections: scanner tag, prompt rule, notice in the
  signals, fragment-level echo check on every output field, system-added
  disclosure with a warning that records what the model itself did.
- Deliberately not built: incremental indexing, conversation memory, an LLM
  reranker or multi-query expansion (each extra call is minutes on CPU),
  a web UI.
- Known limitations: 7B arithmetic (Q3); non-deterministic conflict
  reporting; citation attribution inside the conflicts field; latency on CPU;
  the injection scanner is pattern-based and would miss novel phrasings.

## 12. Where things are

- `SETUP.md` — fresh-machine instructions, verified commands and timings.
- `NOTES.md` — raw working notes per step, all measurements.
- `eval/results/` — four committed runs (Markdown + JSON).
- `logs/` (git-ignored) — application log and per-question JSON lines.
- Commit history — one commit per step and per evaluation round.

Remaining plan steps: Step 10 (optional 7B vs 3B comparison; a first data
point showed the 3B model answering correctly but citing weaker sections)
and Step 11 (full README with architecture diagram, weaknesses, omissions,
and the closing thought).
