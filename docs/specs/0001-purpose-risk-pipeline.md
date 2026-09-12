## Problem Statement

iDISK 2.0 leaves `Purpose` and `Risk` empty on all 92,442 DSLD-only DSP rows,
because it takes both fields from LNHPD alone. US supplement labels carry no
purpose field, so the information exists only as free prose inside a label's
`statements[]` — a sleep aid's purpose may sit in a `General Statements: All
Other Content` block between "Made in the USA" and "ACTUAL SIZE".

`docs/dsld-purpose-risk-findings.md` established that the information is
recoverable (purpose explicit on ~68% ±8pp of sampled labels, risk on ~58%),
that a statement-type→column mapping cannot get at it, and that every candidate
skip filter leaks. ADR 0001 then settled the pipeline design.

What exists today is an extraction prompt, a JSON schema, a grounding check, a
frozen 1,000-label pilot and a 100-label gold set with 336 hand-verified
evidence spans. What does not exist is any way to run it: the only `extract()`
is Anthropic-only and this machine has no `ANTHROPIC_API_KEY`. There is no
local inference path, no results store, no scorer, and no CLI.

So a researcher who wants to recover these two fields has a proven method, a
reference set to score against, a 27B model sitting on disk — and no way to
turn any of it into rows. Worse, the model family in question has a known CUDA
bug that makes it load normally and emit fluent garbage, so even an ad-hoc
script could produce 92,442 plausible, entirely fabricated rows with nothing
in the loop to notice.

## Solution

A local extraction pipeline that runs the existing prompt and schema through
the on-disk `Qwen3.8-27B` Q6_K model, writes one reviewable row per label into
a timestamped run directory, and scores any run against the gold set.

The researcher edits one prompt file, runs a small batch, and reads the prompt
diff printed above the metric delta. When the prompt is good, the same function
runs the full corpus over several days, resumable across crashes, with a
running grounding rate on stdout as a live corruption alarm. Every batch is
preceded by a `smoke` check that puts five gold labels through the real model
and refuses to proceed unless evidence spans come back verbatim.

Nothing in the design pretends the agent path can be automated or that the API
path can run on this machine. Three drivers write the same row shape; only the
loop differs, and `provenance` records which one wrote each row.

## User Stories

1. As a researcher, I want to extract purpose and risk from a DSLD label using a model that runs on this machine, so that I am not blocked on an API key that does not exist here.
2. As a researcher, I want the local driver to be the production path, so that the full corpus run does not depend on external service availability or cost.
3. As a researcher, I want an `api` driver writing the same row shape, so that the pipeline still works on a machine that does have an Anthropic key.
4. As an agent working in-session, I want to write conforming rows from a subagent fan-out, so that a human-directed labelling pass lands in the same store as scripted runs.
5. As a researcher, I want `provenance` recorded as a column on every row, so that I can tell which driver produced a result without it branching the logic that produced it.
6. As a researcher, I want `latency_ms` to be nullable, so that the agent driver can leave it empty rather than report a number not comparable to inference latency.
7. As a researcher, I want one shared validation function that every driver's rows must pass, so that the contract is enforced rather than merely documented.
8. As a researcher, I want each extracted claim checked for whether its evidence appears verbatim in the passage, so that fabrication is caught without a second model or a human read.
9. As a researcher, I want the grounding check to be whitespace-insensitive, so that a model faithfully quoting across a line break is not failed for formatting.
10. As a researcher, I want a self-reported confidence capped when evidence is ungrounded, so that a confident hallucination cannot score itself highly.
11. As a researcher, I want an empty extraction to score zero confidence, so that "found nothing" is never reported as a confident answer.
12. As a researcher, I want structured output enforced by a GBNF grammar generated from the same JSON schema the prompt describes, so that the grammar cannot silently desync from the schema.
13. As a researcher, I want a startup check asserting fixture strings are accepted and rejected as expected, so that a silently permissive schema→grammar converter is caught before a batch runs.
14. As a researcher, I want post-hoc validation retained on top of the grammar, so that shape guarantees are never mistaken for truthfulness guarantees.
15. As a researcher, I want a `smoke` command that puts real gold labels through the real prompt and grammar, so that a corrupted inference path is detected by something a 2-token "ok" generation cannot fake.
16. As a researcher, I want `smoke` to assert a minimum grounding rate before every batch, so that a ggml upgrade that silently corrupts the DeltaNet path cannot quietly poison a multi-day run.
17. As a researcher, I want the running grounding rate reported during a long run, so that corruption appearing mid-run is visible rather than discovered at the end.
18. As a researcher, I want results written as JSONL under a timestamped run directory, so that the store follows the convention the DSLD census and pilot runs already use.
19. As a researcher, I want the manifest committed and raw payloads gitignored, so that the reproducible record survives while large payloads stay out of git.
20. As a researcher, I want each row to carry `config_hash` and `provenance` inline, so that a row is interpretable without joining it back to a manifest.
21. As a researcher, I want runs loadable into pandas by globbing run directories, so that comparing runs needs no database.
22. As a researcher, I want a config hash over prompt, model, decode params, schema and renderer version, so that two runs pool only when their output is genuinely comparable.
23. As a researcher, I want `enable_thinking: false` recorded explicitly in the decode config, so that a silent flip of the model's default thinking mode cannot make results incomparable without anyone noticing.
24. As a researcher, I want a hand-bumped `renderer_version` beside the renderer, so that editing what the model sees invalidates the hash even when prompt, model and params are unchanged.
25. As a researcher, I want the git commit recorded in the manifest and excluded from the hash, so that two genuinely identical runs a week apart still pool.
26. As a researcher, I want the resolved config stored verbatim beside its hash, so that a hash is always invertible into the thing it names.
27. As a researcher, I want a single editable prompt file per step with no version tree, so that a dozen dead prompt files do not accumulate during iteration.
28. As a researcher, I want each run's manifest to embed the prompt text verbatim, so that past runs stay interpretable even though iteration happens uncommitted and dead prompts get deleted.
29. As a researcher, I want a free-text note per run, so that the manifest records what I was trying to achieve, not only what changed.
30. As a researcher, I want a comparison helper that prints the prompt diff above the metric delta, so that I read what changed rather than only that something changed.
31. As a researcher, I want the model loaded in-process rather than behind a server, so that the runtime proven correct by measurement is the one that actually runs.
32. As a researcher, I want a status on every row distinguishing `ok`, `empty`, `unparseable` and `error`, so that "correctly found nothing" is never conflated with "the machinery broke".
33. As a researcher, I want `empty` treated as a first-class success, so that the 28% of labels with no purpose and 29% with no risk are recorded as correct results.
34. As a researcher, I want raw model text kept on every row regardless of status, so that an unparseable result can be diagnosed after the fact.
35. As a researcher, I want no inline retries, so that the failure rate stays visible as the early signal that a prompt or grammar is wrong.
36. As a researcher, I want a retry to be an explicit second run over non-`ok` IDs with its own run directory, so that retried work is attributable rather than blended into the original run.
37. As a researcher, I want latency recorded per row, so that throughput projections rest on measurements rather than estimates.
38. As a researcher, I want a loader handing records to N workers with a single appender writing rows, so that concurrent workers never corrupt the output file.
39. As a researcher, I want resume computed as input IDs minus IDs already present in the output, so that no separate progress file can contradict the data.
40. As a researcher, I want a malformed trailing line discarded and its record re-run, so that a crash mid-write costs one record rather than corrupting a run.
41. As a researcher, I want `--resume <run_id>` to be explicit and never automatic, so that appending to an existing run directory is always a deliberate act.
42. As a researcher, I want each resume's start time recorded in the manifest, so that an appended-to run directory still reads as an honest record.
43. As a researcher, I want a poison guard that writes a record as `poison` after repeated hand-outs, so that one pathological label cannot stall a multi-day job across restarts.
44. As a researcher, I want progress reported as plain appended stdout lines on a timer, so that a notebook cell does not fill with thousands of ANSI redraw artefacts.
45. As a researcher, I want progress to report rate over a trailing window rather than cumulatively, so that a slowdown is visible rather than averaged away.
46. As a researcher, I want progress to report elapsed, done, remaining, projected finish and the status breakdown, so that I can decide whether to let a multi-day run continue.
47. As a researcher, I want a headline score of per-field presence and absence over the gold set's 200 cells, so that the primary metric is robust at this sample size.
48. As a researcher, I want hallucination rate called out separately, so that the stated priority of grounding over recall is measured directly.
49. As a researcher, I want evidence-span precision and recall per claim as a second number, so that a coarse headline is not the only view.
50. As a researcher, I want the detail metric's ceiling documented in the scorer itself, so that the 29% of gold risk entries sharing an evidence block is read as a metric limitation, never as model error.
51. As a researcher, I want the gold set's status as one model's reading rather than human adjudication to be stated wherever scores are reported, so that agreement is not mistaken for correctness.
52. As a researcher, I want to run the pipeline over a small batch by ID or limit, so that prompt iteration costs minutes rather than days.
53. As a researcher, I want to score any run directory against the gold set with one command, so that the iteration loop is edit, run, score, read.
54. As a researcher, I want a `stats` command summarising a run, so that I can see status breakdown and grounding rate without writing pandas each time.
55. As a researcher, I want `run()` to be a plain importable function returning the run directory, so that a notebook and the CLI call one implementation.
56. As a researcher, I want the CLI to contain argument parsing and no logic, so that behaviour cannot diverge between the two callers.
57. As a researcher, I want notebooks to import and call the pipeline rather than shell out, so that a notebook never reimplements a stage.
58. As a researcher, I want long jobs to live outside notebooks, so that a 14-day run does not depend on a kernel surviving, and is not exposed to the stale-editor-tab clobbering this repo has already seen.
59. As a researcher, I want a notebook to show live progress and the final run directory path, so that inspecting a run means reading its JSONL and manifest.
60. As a researcher, I want the scorer covered by fixture tests, so that a broken scorer cannot yield plausible numbers and misdirect a week of prompt tuning.
61. As a researcher, I want validation covered by fixture tests, so that grounding, the confidence cap and status assignment are verified without a GPU.
62. As a researcher, I want throughput measured on small batches before the full corpus runs, so that a multi-day job starts once with known settings.
63. As a researcher, I want concurrency treated as an open measurement rather than an assumption, so that this hybrid architecture's per-stream recurrent state is characterised rather than guessed at.
64. As a researcher, I want the full corpus run once and last, so that days of GPU time are not spent on a configuration still being tuned.
65. As a maintainer, I want extraction to stay on the existing DSLD label loading path, so that the pipeline reads labels the same way the pilot and gold set did.
66. As a maintainer, I want the renderer to remain the single place deciding what the model sees, so that `SKIP_TYPES` and the ingredient cap have one home.
67. As a researcher, I want gold set expansion to draw randomly from the remaining 900 pilot labels, so that the age-biased lowest-ID decile is not compounded.

## Implementation Decisions

All of the following are settled in ADR 0001; this spec restates them as build
instructions and does not reopen them.

**Module layout.** A new `engine/pipeline` package with one module per stage:
a renderer (passage from label), an extractor (one record to one result), a
validator (raw to validated rows), a scorer (rows plus gold to metrics), a run
orchestrator, and a CLI. The CLI is argparse only and holds no logic.

**Three drivers, one row contract.** The enforced abstraction is the output row
plus the validation function, not a `Backend` class. `local` (in-process
llama.cpp, the production path), `api` (Anthropic, needs a key absent here) and
`agent` (an in-session procedure fanning out subagents) each own their own loop
and write conforming rows. A uniform `Backend` interface is explicitly
rejected: subagents can only be spawned from inside an agent's turn, so an
`AgentBackend` would raise `NotImplementedError` when driven from a script, and
an interface that lies is worse than two entry points. Prompt files, schema,
grammar, grounding check, confidence cap, results store and scorer are shared;
`provenance` is a column, never a branch.

**Reuse of existing extraction assets.** The prompt text, JSON schema, `Entry`
and `Extraction` shapes, `_normalise` grounding, `UNGROUNDED_CONFIDENCE_CAP`
and `SKIP_TYPES` already exist in the label purpose/risk extraction service and
are moved or re-exported rather than rewritten. The Anthropic `extract()`
becomes the `api` driver; it is not deleted.

**Results store.** JSONL under timestamped run directories following the
existing convention: manifest committed, raw payloads gitignored, one row per
attempt. Stats are a glob into pandas — at ~92k rows this is tens of MB.
SQLite is rejected (breaks the convention, not human-readable, and the agent
driver would have to issue SQL in-session rather than append reviewable dicts),
as is a JSONL-plus-derived-SQLite index (a stale index is a silent wrong-answer
source). Revisit only when a query is actually painful.

**Config identity.** A sha256 over canonical JSON of prompt id, prompt sha,
model id, decode params, schema sha and renderer version:

```
config_hash = sha256(canonical_json({
    prompt_id, prompt_sha, model_id, decode, schema_sha, renderer_version
}))
```

`decode` includes `enable_thinking: false` explicitly, because the model ships
with thinking on and a silent flip would make every result incomparable.
`renderer_version` is a hand-bumped integer beside the renderer — hashing the
function source would churn on comment edits, hashing its output would require
running it first. The git commit goes in the manifest, not the hash, so two
identical runs a week apart still pool. The resolved config blob is stored
verbatim beside its hash.

**Prompt versioning.** One editable file per prompt step, no version tree. Each
manifest embeds the prompt text verbatim plus its sha and a free-text note.
Numbered files accumulate dead files and, once deleted, orphan every past run's
`prompt_sha`; any scheme relying on `git show` has holes precisely during the
uncommitted iteration phase that matters most. Comparison is a helper printing
the prompt diff above the metric delta.

**Model and runtime.** `Qwen3.8-27B` Q6_K (16.4 GiB, an unextensioned blob in
the HF cache, which is why earlier `*.gguf` searches missed it), loaded fully
offloaded in-process via `llama-cpp-python`. Q6_K over the usual Q4_K_M is a
benefit here: the grounding check is a literal substring match, and
verbatim-copying fidelity is what aggressive quantisation degrades first.
`llama-server` is not used. The in-process path was chosen by measurement, not
assumption: five gold labels through the real prompt and grammar returned 43/43
entries verbatim-grounded with the GPU verified exclusive.

**Structured output.** GBNF generated from the one schema via
`LlamaGrammar.from_json_schema`, verified working on the existing nested
arrays-of-objects schema. Hand-writing a grammar is rejected as a second source
of truth whose desync produces plausible, parseable, wrong-shaped output —
the worst failure mode, because nothing errors. A startup check asserts fixture
strings are accepted or rejected as expected (valid object; missing `evidence`;
wrong type on `purpose_confidence`; empty arrays). Post-hoc validation is
retained on top: a grammar guarantees shape, never truthfulness.

**Failure handling.** `status ∈ {ok, empty, unparseable, error}`, plus `poison`
from the durability guard. Raw text always kept. `empty` is a first-class
success — the gold set says no purpose is correct on 28% of labels and no risk
on 29%, and conflating `empty` with `unparseable` would hide exactly what the
smoke check exists to catch. No inline retries: they hide the failure rate and
corrupt latency stats. A retry pass is an explicit second run over non-`ok` IDs
with its own run directory.

**Durability.** A central loader owns the queue and hands out records; N
workers take, render, call, and hand results to a single appender thread that
flushes per row. Records are independent, so there is no cross-worker
coordination. Resume is input IDs minus IDs present in the output; a malformed
trailing line is discarded and that record re-run. `interrupted` is not a
stored status — recording work-in-flight needs two writes and mutable rows, and
a crash kills the very code that would write it, while absence already means
"not done, for whatever reason" at a cost of ~15s to redo. A sidecar progress
file is rejected: two files can disagree, and a checkpoint contradicting the
data is worse than none. A poison guard counts attempts per ID in memory and
writes the record as `poison` after the second or third hand-out.

**Progress reporting.** Plain appended stdout lines on a ~60s timer, never ANSI
carriage-return redraw, which becomes thousands of garbage lines in a notebook
cell. Each line reports elapsed, done, remaining, rate over a trailing window,
projected finish, status breakdown, and the running grounding rate.

**Scoring.** Headline is per-field presence/absence over 200 cells (100 labels
× 2 fields) with hallucination rate called out separately. Detail is
evidence-span precision and recall per claim, matching on overlap of verbatim
spans; its ceiling is documented in the scorer's docstring, since 55/190 gold
risk entries share an evidence block with a sibling. LLM-as-judge is rejected
(with no API key the judge is the same local model grading itself, or a
human-directed session; both unreproducible). Token overlap on `text` is
rejected — it cannot distinguish a missed claim from a rephrased one, and those
demand opposite prompt changes. 100 labels gives ±10pp at best: the gold set is
a screen for breakage and large wins, and when two prompts land within a few
points the answer is more gold labels drawn randomly from the remaining 900,
not a finer metric on the same 100.

**Throughput.** Measured single-stream 11.6–54.6s per label at ~15 tok/s;
median label ≈ 176–250 output tokens ≈ 12–16s, so 92,442 records is roughly
14–18 days of continuous GPU time. Tune prompt and concurrency on small
batches; run the full corpus once, last. Splitting purpose and risk into two
calls is rejected at 27B (roughly doubles cost for accuracy the probe suggests
is not needed; risk overlap already 0.88–0.95). Evidence as character offsets
is rejected — it would break the grounding check that proved the runtime
honest. Reducing the corpus to a stratified sample is rejected. Whether
batching 4–8 streams gives 3–5× is open as a measurement, not a decision: this
hybrid keeps recurrent state per stream rather than a plain KV cache, so
batching behaviour is not predictable from first principles.

**CLI surface.** Subcommands `smoke`, `run`, `score`, `stats`. `run()` is a
plain function returning the run directory; the CLI parses flags and calls it,
and a notebook imports and calls the same function.

## Testing Decisions

**What makes a good test here.** Test external behaviour at a module's
interface, not its internals. A test should state a fact about the pipeline
that a reader would want to be true — "ungrounded evidence caps the confidence",
"a correctly-empty extraction scores as a hit, not a miss" — and should keep
passing through a refactor that preserves those facts. No test asserts on
private helpers, call counts, or intermediate data structures.

**Three seams, confirmed with the developer.** ADR 0001 §12 names six modules;
only three are seams, and two of those carry fixture tests.

1. **Validation** — raw model dict plus passage in, validated rows out. A pure
   function with no GPU and no network, and the highest deterministic seam in
   the pipeline. Grounding, the confidence cap and status assignment all land
   here. Fixtures: grounded evidence; evidence differing only in whitespace and
   line breaks; evidence absent from the passage; empty arrays; a wrong-typed
   confidence; raw text that does not parse.

2. **Scoring** — rows plus gold in, metrics out. ADR 0001 §12 puts this in
   scope explicitly despite the repo having no test suite, because a broken
   runtime is caught by `smoke` while a broken scorer silently yields plausible
   numbers and misdirects a week of prompt tuning. Fixtures named in the ADR:
   exact match, paraphrase, missed claim, hallucinated claim, correctly empty.
   Add one for the shared-evidence-block case, asserting the documented ceiling
   behaves as documented rather than as a silent penalty.

3. **Smoke** — a live assertion, not a unit test. Five gold labels through the
   real model, prompt and grammar, asserting a minimum grounding rate. It
   cannot be replaced by fixtures: it exists precisely to detect a runtime that
   loads normally and emits fluent garbage, which every mock would hide. It
   runs before every batch and is part of the product, not the test suite.

**Deliberately not seams.** The renderer is covered transitively — its output
is the passage validation grounds against, and `renderer_version` is in the
config hash, so a change is never silent. The extractor and run orchestrator
are the GPU and IO loop; they are covered by `smoke` plus the fact that
resume-by-absence is directly observable in a run directory. Adding mocks there
would test the mock.

**Prior art.** The repo has no test suite yet, so these tests establish the
pattern: plain `pytest` under a top-level `tests/` directory, `pytest` already
declared in the dev extra. The nearest existing prior art for fixture-shaped
verification is the pilot notebook's checks — gold-is-a-subset-of-pilot, the
`SKIP_TYPES` assertions, the boilerplate-leak count. Those are the right kind
of assertion in the wrong place; the scorer and validator tests move that style
into something that runs without a human reading output.

**Gold set fixtures come from the real gold set.** Test fixtures are small
hand-written dicts, not slices of `datasets/pilot/gold-100.jsonl`, so that a
future regeneration of the gold set does not break the scorer tests. The gold
file itself is exercised by `smoke` and by `score`.

## Out of Scope

- **Running the full 92,442-record corpus.** The pipeline must be able to, and
  throughput work is in scope, but the multi-day run itself is a separate
  operational act performed once, last, after prompt iteration has settled.
- **Deciding the concurrency level.** ADR 0001 §9 leaves batching 4–8 streams
  open as a measurement. Building the pipeline so `n_workers` is a parameter is
  in scope; choosing its value is not.
- **The boilerplate-suppression pass.** Per-entry rows were chosen so such a
  pass has somewhere to act, but that pass is still undesigned.
- **Expanding the gold set beyond 100 labels.** The decision rule is recorded
  (draw randomly from the remaining 900, not the age-biased lowest-ID decile),
  but the work is triggered only when two prompts land within a few points.
- **Writing recovered Purpose/Risk back into DSP.csv or the knowledge graph.**
  This spec ends at scored rows in a run directory.
- **Validating against the 208 `both` rows.** Those carry curated LNHPD
  Purpose/Risk alongside a DSLD label and are deliberately held back as the
  only independent check; spending them here would leave nothing to validate
  against.
- **Prompt content iteration.** The pipeline makes iteration cheap; which
  prompt wins is downstream work.
- **A server-based runtime.** `llama-server` is explicitly not used, and no
  rebuild of the local llama.cpp checkout is needed.
- **Testing the renderer, extractor or run loop directly.** Covered
  transitively or by `smoke`, per the seam decision above.

## Further Notes

**The silent-corruption risk is the reason for several otherwise-odd choices.**
Research flagged a CUDA bug in this architecture's DeltaNet path that made
models load normally and emit fluent garbage, with a stale `libggml-cuda.so`
keeping the bug alive after a rebuild. This machine has two disagreeing ggml
copies — a pip wheel and a local llama.cpp checkout. That is why `smoke` asserts
grounding rather than parseability, why `empty` must not be conflated with
`unparseable`, and why the running grounding rate is on the progress line.
Anyone tempted to simplify those three should read ADR 0001 §7 first.

**The gold set is a reference, not ground truth.** It is one frontier model's
reading of the label, auditable because evidence spans are kept, not a human
adjudication. Precision against it means agreement with that reading. It is
also the first 100 rows of an id-sorted pilot file — the lowest-numbered decile
— so it is age-biased, though passage length and statement count are close to
the remaining 900.

**Two claims in the findings doc are already corrected by ADR 0001** and should
not be re-derived: the weights are on disk (they were never pending a
download), and splitting purpose and risk into two calls was written for a 7–8B
model and is rejected at 27B.

**Run directories are appended to across restarts**, in mild tension with the
"immutable run dir" convention the census and pilot runs follow. This is
accepted and mitigated by making `--resume` explicit and recording each
resume's start time in the manifest.

**`gh` was not installed on the development machine** when this spec was
written; it has since been installed to `~/.local/bin` but is not yet
authenticated.

---

Derived from [ADR 0001](../adr/0001-local-llm-purpose-risk-pipeline.md) and
[the findings doc](../dsld-purpose-risk-findings.md). Where this spec and the
ADR appear to differ, the ADR is authoritative.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
