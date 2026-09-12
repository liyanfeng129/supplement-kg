# 1. Local LLM pipeline for DSLD purpose/risk extraction

- **Status:** Accepted
- **Date:** 2026-09-11
- **Supersedes / corrects:** two stale claims in `docs/dsld-purpose-risk-findings.md` §6 and §10 (see [Corrections](#corrections))

## Context

`DSP.csv` has `Purpose` and `Risk` populated only where a row carries an
LNHPD `NHP_ID`. All 92,442 DSLD-only rows are empty by construction — a
regulatory artefact, not a data bug. US supplement labels have no purpose
field, so the text exists only inside `statements[]` as free prose.

`docs/dsld-purpose-risk-findings.md` established that the information is
recoverable (purpose explicit on ~68% ±8pp of sampled labels, risk on ~58%),
that a statement-type→column mapping cannot get at it (58% of purposes live in
the `General Statements: All Other Content` junk drawer), and that every
candidate skip filter leaks. It concluded: scan every label, one merged passage
per label, LLM extraction.

What existed before this decision: a prompt and JSON schema
(`engine/services/extraction/label_purpose_risk.py`), an Anthropic-only
`extract()` that **cannot run on this machine** (no `ANTHROPIC_API_KEY`), a
frozen 1,000-label pilot, and a 100-label gold set with 336 hand-verified
evidence spans. No local inference path, no results store, no scorer, no CLI.

This ADR records the pipeline design that closes that gap.

## Decision

### 1. One row contract, three drivers — not one driver with three backends

The shared, enforced abstraction is **the output row plus the validation
function**, not a `Backend` class. Three drivers write conforming rows:

| Driver | Loop owner | Availability |
|---|---|---|
| `local` | scripted, in-process llama.cpp | works today; the production path |
| `api` | scripted, Anthropic | needs a key that does not exist here |
| `agent` | an in-session Claude Code procedure fanning out subagents | live session only |

Rejected: a uniform `Backend` interface with an `agent` implementation.
Subagents can only be spawned from inside an agent's turn — there is no API a
script can call — so an `AgentBackend` would raise `NotImplementedError` when
driven from a loop. An interface that lies is worse than two entry points.

Consequence: prompt files, JSON schema, GBNF grammar, grounding check,
confidence cap, results store and scorer are **shared by all three**. Only the
loop differs. `provenance` is a column, never a branch in logic. `latency_ms`
is nullable — the agent driver leaves it null rather than reporting a number
that is not comparable to inference latency.

### 2. Results store: JSONL run directories

Follows the existing convention (`data/dsld/runs/<TIMESTAMP>Z/`, manifest
committed, `raw/*.jsonl` gitignored). Every row carries `config_hash` and
`provenance` inline. Stats are a `load_runs()` glob into pandas; at ~92k rows
this is tens of MB, not a scale problem.

Rejected: SQLite. Faster to query, but it breaks the run-dir convention, is
not human-readable, and the agent driver would have to issue SQL inserts
in-session rather than appending reviewable dicts. Deferred until a query is
actually painful. Also rejected: JSONL-plus-derived-SQLite-index — a stale
index is a silent wrong-answer source.

### 3. Config identity: hash over everything that changes the output

```
config_hash = sha256(canonical_json({
    prompt_id, prompt_sha, model_id, decode, schema_sha, renderer_version
}))
```

`decode` includes `enable_thinking: false` explicitly — Qwen3.8 ships with
thinking on (`reasoning_effort: xhigh`), and a silent flip would make every
result incomparable.

`renderer_version` is a hand-bumped integer beside `label_text()`. The
renderer decides what the model sees — `[Statement Type]` prefixes, the
25-ingredient cap, `SKIP_TYPES` — so editing it changes every output while
prompt, model and params are unchanged. This repo has already edited
`SKIP_TYPES` once and reverted it. Hashing the function source would churn on
comment edits; hashing its output would require running it first.

The git commit goes in the **manifest, not the hash**: including it would give
two genuinely identical runs different hashes a week apart and stop them
pooling. The resolved config blob is stored verbatim next to its hash, so a
hash is always invertible.

### 4. Prompt versioning: one working file, text embedded per run

One editable file per prompt step (`prompts/purpose_risk.md`). **No version
tree.** Each run's manifest embeds the prompt text **verbatim** plus its sha
and a free-text `note` ("tightened the boilerplate rule").

The driving requirement: most prompts are throwaway — many iterations, one
survivor — and iteration happens *uncommitted*. Numbered files (`v1.md`,
`v2.md`) accumulate a dozen dead files and, once deleted, orphan every past
run's `prompt_sha`. Any scheme relying on `git show` to recover prompt text has
holes precisely during the phase that matters most.

Comparison is therefore a helper that prints the **prompt diff above the
metric delta** — the thing actually worth looking at. A sha says *that* it
changed, the diff says *what*, the note says what you were trying to achieve.

### 5. Model: `Qwen3.8-27B`, Q6_K, in-process

On disk at
`/var/tmp/ge74wan/hf_cache/hub/models--unsloth--Qwen3.8-27B-GGUF/blobs/3f22707…`
— 16.4 GiB, unextensioned blob (which is why earlier `*.gguf` searches missed
it). `general.file_type = 15` = **Q6_K**, Unsloth imatrix (1,251 chunks).

Architecture is `qwen35`: **hybrid** — 65 blocks, full attention every 4th
layer, Gated DeltaNet linear attention elsewhere, GQA 24/4, 256K native
context, MTP head. Loads fully offloaded in 3.1s on the RTX 3090 Ti (24 GiB),
leaving ~7.5 GiB.

Q6_K rather than the Q4_K_M usually recommended for 24 GiB is a **benefit
here**: the grounding check is a literal substring match, and verbatim-copying
fidelity is exactly what aggressive quantisation degrades first.

### 6. Structured output: GBNF generated from the one schema

`LlamaGrammar.from_json_schema(SCHEMA)` — verified working on the existing
nested arrays-of-objects schema, which the pilot notebook listed as an open
question. Hand-writing a grammar is rejected: it is a second source of truth,
and a desynced grammar produces plausible, parseable, wrong-shaped output —
the worst failure mode, because nothing errors.

A startup check asserts fixture strings are accepted/rejected as expected
(valid object; missing `evidence`; wrong type on `purpose_confidence`; empty
arrays), guarding against the converter being silently permissive.

The existing post-hoc validation is **retained on top**: `_normalise`
whitespace-insensitive substring grounding and the `UNGROUNDED_CONFIDENCE_CAP`
floor. A grammar guarantees *shape*, never *truthfulness*. Both are needed.

### 7. Runtime: in-process `llama-cpp-python` — proven, not assumed

Research flagged a CUDA bug in the DeltaNet path that made these models **load
normally and emit fluent garbage** (broken at `221f0f6`, fixed by `ece963f`,
~build 10450+), with a stale `libggml-cuda.so` keeping the bug after a
rebuild. This machine has two disagreeing ggml copies: the pip wheel (0.20.0)
and `/var/tmp/ge74wan/llama.cpp` (0.22.0).

A 2-token "ok" generation cannot distinguish a correct model from a corrupted
one. So the runtime was chosen by **measurement**: five gold labels through the
real prompt and grammar, scoring whether `evidence` spans appear verbatim in
the passage.

**Result: 43/43 entries verbatim-grounded (100%)**, GPU verified exclusive
(1 MiB used, no compute apps). A corrupted inference path cannot reproduce 43
exact substrings from 1,600–4,000-char passages. The in-process path is sound;
no server rebuild is needed, and `llama-server` is not used.

This check becomes the permanent `smoke` command, asserting a **minimum
grounding rate** before every batch. On this architecture silent corruption is
a live risk across any ggml upgrade, and grounding — not parseability — is the
only cheap thing that detects it.

### 8. Failure handling: one row per attempt, with `status`

`status ∈ {ok, empty, unparseable, error}`. Raw text is always kept.

`empty` is a **first-class success**: the gold set says no purpose is correct
on 28% of labels and no risk on 29%. Conflating `empty` with `unparseable`
would make it impossible to tell "correctly found nothing" from "machinery
broke" — and a corrupted runtime produces both, hiding exactly what §7 exists
to catch.

No inline retries: they hide the failure rate, which is the primary early
signal that a prompt or grammar is wrong, and they corrupt latency stats. A
retry pass is an explicit second run over `status != 'ok'` IDs with its own run
dir.

### 9. Throughput: optimise first, full corpus last

Measured single-stream: 11.6s–54.6s per label, scaling with output tokens
(~15 tok/s). Median label ≈ 176–250 output tokens ≈ 12–16s.

**92,442 records ≈ 14–18 days of continuous single-stream GPU time.**

Plan: tune prompt and concurrency on small batches; run the full corpus once,
last. Multi-day is acceptable.

Rejected: **splitting purpose and risk into two calls** — §6 of the findings
doc suggested it for a 7–8B model; at 27B it roughly doubles cost for accuracy
the probe suggests is not needed (risk overlap already 0.88–0.95). Rejected:
**evidence as character offsets** to shorten output — it would break the
grounding check that proved the runtime honest. Rejected: **reducing the
corpus** to a stratified sample.

Open as a *measurement*, not a decision: whether batching 4–8 streams gives
3–5×. This hybrid keeps recurrent SSM state per stream rather than a plain KV
cache, so batching behaviour is not predictable from first principles. Measure
it.

### 10. Durability: one row per record, resume by absence

A central loader owns the queue and hands out the next record on request; N
workers each loop *take → render → call → hand result to the appender*.
Records are independent, so no cross-worker coordination. One appender thread
flushes per row (free at 15s/record); workers never write concurrently.

**Resume = input IDs minus IDs present in the output.** A malformed trailing
line is discarded and that record re-run.

`interrupted` is **not a stored status**. Recording work-in-flight would
require writing before the call and amending after — two writes and mutable
rows, against the append-only choice — and a crash kills the very code that
would write it. Absence already means "not done, for whatever reason", and
re-running one record costs ~15s.

Rejected: a sidecar `progress.json`. Two files can disagree, and a checkpoint
that contradicts the data is worse than none. The data is the single source of
truth; a crashed run's progress remains derivable from its JSONL.

**Poison guard:** attempts per ID are counted in memory per run; after the
second or third hand-out the record is written as `status: poison` and skipped,
so one pathological label (the corpus max is 8,960 chars) cannot stall a
multi-day job across restarts.

**Progress reporting** goes to stdout as plain appended lines on a ~60s timer —
no ANSI carriage-return redraw, which would become thousands of garbage lines
in a notebook cell. Each line reports elapsed, done, remaining, rate over a
*trailing* window (cumulative rate hides slowdowns), projected finish, the
`status` breakdown, and the **running grounding rate** — the corruption alarm.

### 11. Scoring: robust headline, detailed second number

- **Headline:** per-field presence/absence over 200 cells (100 labels × 2
  fields), with **hallucination rate** — a purpose asserted where gold says
  none — called out separately. This directly measures the stated priority of
  grounding over recall.
- **Detail:** evidence-span precision/recall per claim, matching on overlap of
  verbatim spans.

The detail metric's ceiling is documented in the scorer's docstring: 55/190
gold risk entries (29%) share an evidence block with a sibling, so span
matching cannot separate three claims citing one WARNING block from one claim
citing it. That 29% is a metric limitation, never model error.

Rejected: **LLM-as-judge** — with no API key the judge would be the same local
model grading itself, or a human-directed session; both are unreproducible and
cannot drive an iteration loop. Rejected: **token overlap on `text`** (used in
the probe as a sanity signal only) — it cannot distinguish a missed claim from
a rephrased one, and those demand opposite prompt changes.

**Honest constraint:** 100 labels gives ±10pp at best. The gold set is a
*screen* that catches breakage and large wins. When two prompts land within a
few points, the answer is more gold labels — drawn **randomly from the
remaining 900**, since the existing 100 are the lowest-ID decile and therefore
an age-biased slice — not a finer metric on the same 100.

### 12. Entry point: importable functions, thin CLI

```
engine/pipeline/render.py     passage from label
engine/pipeline/extract.py    one record -> one result
engine/pipeline/validate.py   raw -> validated rows (grounding, confidence cap)
engine/pipeline/score.py      rows + gold -> metrics
engine/pipeline/run.py        run(config, record_ids, n_workers, resume) -> Path
engine/pipeline/cli.py        argparse only, no logic
```

Subcommands: `smoke`, `run`, `score`, `stats`.

`run()` is a plain function returning the run directory. The CLI parses flags
and calls it; **a notebook imports and calls the same function** — one
implementation, two callers. Notebooks show live progress in the cell and the
final run-dir path, then inspect the run dir's JSONL and manifest to see input
and output. They do not shell out to a subprocess and do not reimplement any
stage.

Notebooks remain what notebook 02 already is: a place to look at things, not a
place where long jobs live. (A 14-day run cannot survive a kernel, and this
repo has had `.ipynb` edits silently clobbered by a stale editor tab.)

**A scorer test is in scope** despite the repo having no test suite. A broken
runtime is caught by `smoke`; a broken scorer silently yields plausible numbers
and misdirects a week of prompt tuning. Fixtures: exact match, paraphrase,
missed claim, hallucinated claim, correctly empty.

## Consequences

**Good**

- The runtime is *proven* correct on a model family with a known
  silent-corruption bug, and `smoke` keeps proving it.
- The GBNF-from-schema result closes the pilot notebook's open question without
  a hand-written grammar.
- Prompt iteration is reproducible and cheap: edit one file, `run --limit 20`,
  `score`, read the prompt diff above the metric delta.
- Nothing in the design pretends the agent path can be automated, or that the
  API path can run here.

**Costs and risks accepted**

- ~15 tok/s single-stream means the full corpus is days, not hours. Concurrency
  is unmeasured on this hybrid architecture.
- The gold set is an **Opus reading of the label**, not human adjudication
  (`gold_set.py:1-21` says so). Precision against it means *agreement with that
  reading*. At 100 labels that is a screen, not a ranking.
- 29% of gold risk entries share evidence blocks, capping claim-level risk
  scoring.
- Run directories are appended to across restarts, in mild tension with the
  "immutable run dir" convention. Mitigated by making `--resume <run_id>`
  explicit, never automatic, with each resume's start time in the manifest.

## Corrections

Two claims in `docs/dsld-purpose-risk-findings.md` are now false:

1. §10: *"which local model and quant. That now waits on downloading weights."*
   The weights are on disk and have been since 2026-08-26 — `Qwen3.8-27B`
   **Q6_K**, 16.4 GiB. The blob has no `.gguf` extension, which is why
   filename searches reported none.
2. §6 note 4: *"Consider splitting purpose and risk into two calls."* Written
   for a 7–8B model; rejected at 27B (§9 above).

Also newly known and not recorded there: **~15 tok/s single-stream → 14–18 days
for the full corpus**, and GBNF-from-schema works on the existing nested
schema.

## Status of the model name

`Qwen3.8-27B` released 2026-08-14 (Apache-2.0, dense 27B) — three generations
past Qwen3, whose dense line topped out at 32B with no 27B. Earlier sessions'
assumption that "27B" must be a mistake was wrong.

## Addendum: corrections from implementation (2026-09-12)

Three things this ADR asserts turned out to be wrong or unreachable when the
pipeline was built. The decisions all stand; these are factual repairs.

### A1. The quant is Q4_K_M, not Q6_K (§5)

`general.file_type = 15` is **Q4_K_M**. llama.cpp prints `file type = Q4_K -
Medium` for this blob, and the snapshot symlink names it
`Qwen3.8-27B-UD-Q4_K_XL.gguf` — an Unsloth imatrix dynamic quant (1,251
chunks). §5's "Q6_K rather than the Q4_K_M usually recommended" reads the same
field as saying its opposite.

This does not change the decision. §5's reasoning was that verbatim-copying
fidelity is what aggressive quantisation degrades first, and that fidelity is
*measured*, not assumed: `smoke` scored 11/11 entries verbatim-grounded on the
real weights, and a 20-label batch ran at 100% grounding over 85 entries. The
argument for a higher quant was always an argument for checking grounding, and
the check passes at the quant actually on disk.

### A2. `enable_thinking: false` is unreachable through the chat API (§3)

`llama-cpp-python` 0.3.35 exposes no `chat_template_kwargs` passthrough on
`create_chat_completion`, so the model template's `enable_thinking` branch —
the only thing standing between a run and `reasoning_effort: xhigh` — cannot be
reached that way. Passing it raises `TypeError`.

The extractor therefore renders the ChatML prompt itself and calls
`create_completion`. `tests/test_prompt_format.py` asserts the rendered string
is **byte-identical** to the model's own Jinja template rendered with
`enable_thinking=False`, so the hand-rolled prompt is checked against the
source of truth rather than against a reading of it. The decode config still
records the flag, and it still changes the config hash.

### A3. ggml's CUDA backend fails to init, and falls back to CPU silently

Not anticipated anywhere in this ADR, and the most dangerous of the three.
`ggml_cuda_init` allocates pinned host memory; that call is refused in this
environment, and ggml treats the failure as fatal for the whole backend:
`llama_supports_gpu_offload()` returns False, **`n_gpu_layers` is silently
ignored, and the model runs on CPU**. Measured at roughly 100x slower — the
first smoke attempt burned 100+ minutes of CPU at 20 GB RSS while the GPU sat
at 1 MiB, producing no error at any layer.

`GGML_CUDA_NO_PINNED=1` is now set at module scope in `engine/pipeline/extract.py`
(it must precede the first `llama_cpp` import), and `check_gpu_offload()` runs
inside `smoke()` so a CPU fallback fails the batch instead of quietly costing
days.

This is the same failure *shape* §7 was written about — loads normally,
produces plausible output, nothing errors — but a different cause, and
grounding does not detect it: a CPU run is perfectly grounded, merely
impossibly slow. Hence a separate assertion rather than a tighter grounding
threshold.
