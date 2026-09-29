# Streaming run: keep the model busy on a corpus larger than memory

- **Status:** proposed
- **Date:** 2026-09-26
- **Scope:** `repositories/dsld/loader.py`, `repositories/dsld/store.py`,
  `controllers/qwen_controller.py`, `models/qwen_model.py`, the purpose/risk service,
  and `configs/qwen.toml`.

Builds on `docs/qwen-model-controller-design.md` and amends it in the places listed
under [Departures](#departures).

## Problem

`QwenController.handle(prompts)` takes the whole work set as a list, plans it, and
blocks until every prompt has an outcome. That is right for 64 labels and wrong for
the corpus:

- **Memory.** Holding every flattened label and its prompt at once does not scale to
  the full DSLD.
- **Idle slots if we chunk.** The obvious fix is to call `handle` once per chunk. But
  `handle` returns only when the slowest prompt in the chunk finishes, so at the end
  of every chunk the slots free one at a time and nothing refills them. That is the
  barrier-batch pattern already measured and rejected: refilling a freed slot at once
  was 1.38x faster.
- **Resume granularity.** Outcomes come back only at the end of `handle`, so nothing
  reaches the store until a whole chunk is done.

## The rule

**The local LLM is the bottleneck. Its queue must never be empty while labels remain.**

Everything else — loading, building the prompt, checking, saving — exists to feed the
model and clear its output. Each of those steps must either be cheap (milliseconds
against a generation's seconds) or run off the loop that feeds the model.

## Shape

```
loader.pending(run) ──lazy──► service: label → Prompt ──lazy──► controller.stream()
                                                                   │  submit
                                                                   ▼
                                                           model queue ──► n_parallel slots
                                                                   │  future done
                                                                   ▼
                                                          settle (checks, repair)
                                                                   │  (key, outcome)
                                                                   ▼
                                                  service ──► sink.add(row) ── flushes in batches
```

Everything left of the controller is a lazy iterator: a label is read and wrapped in a
prompt only when the controller pulls it. The loader already hands back cleaned,
flattened labels — the frozen passages the gold set was labelled against — so there is
no cleaning step in the run. The controller keeps a fixed number of
prompts in flight, which is more than the number of slots, so a slot that frees
always finds a prompt already queued in the model. Slots go idle only when the
loader has nothing left.

## Changes, layer by layer

### `repositories/dsld/loader.py` — `pending(...)`

```python
def pending(self, run, path=None, ids=None, limit=None, order=None, seed=None) -> Pending
```

- Takes the same selection arguments as `load`, plus the run directory to resume.
- Works out the id list up front (ids are small) and subtracts `store.done_ids(run)`.
- **Labels are read lazily**: iterating a `Pending` reads one cleaned, flattened
  `Label` per step, ready to go into a prompt. Memory holds the ids, not the labels.
  This means streaming the flattened JSONL line by line rather than going through
  `load_flattened`'s whole-file cache.
- Passages are never re-rendered here, for the same reason as in `load_flattened`:
  the gold set was labelled against these exact strings.
- `Pending` carries `total` (labels still to do), `n_already_done`, `missing_ids` and
  `source`, as `Selection` does, so progress has a denominator and the run can report
  what it left out.

This is where resume lives, per the agreed architecture. There is no progress file.

### `repositories/dsld/store.py` — `Sink`

```python
class Sink:
    def __init__(self, run, flush_every: int = 64, flush_seconds: float = 30.0): ...
    def add(self, row: dict) -> None: ...     # append to buffer; flush if a threshold is hit
    def flush(self) -> None: ...              # append buffered rows to the run's JSONL, fsync
    def close(self) -> None: ...              # final flush
```

- `add` never does more than append to a list and check two thresholds.
- `flush` appends and does not rewrite, so the torn-trailing-line rule in `read_rows` and
  `repair` still holds. `repair(run)` runs once when the sink opens on an existing run,
  for the same reason it exists today.
- It is a context manager, so `close()` runs on normal exit, on an exception, and on
  Ctrl-C.
- **Flushes run on the caller's thread in v1.** A local JSONL append of 64 rows is far
  shorter than the time the queued prompts take to run (see [Sizing](#sizing)), so the
  model does not starve during a flush. If the acceptance measurement ever shows
  starvation (for example, a slow network store), move the flush to a sink-owned
  thread. The controller and service do not change.

### `controllers/qwen_controller.py` — `stream(...)`

```python
def stream(
    self,
    work: Iterable[tuple[K, Prompt]],
    checks: Iterable[Check] = (),
    total: int | None = None,
    on_progress: ProgressFn | None = None,
) -> Iterator[tuple[K, Outcome]]
```

- `K` is opaque to the controller. The service passes the DSLD id, and the controller
  hands it back with the outcome. The controller never learns what a label is.
- **Loop:**
  1. `model.ready()`, then `model.cache_shared_prefix(first)`, where `first` is the
     first `in_flight` prompts pulled from `work`. This runs once per run.
  2. Submit those prompts.
  3. `wait(FIRST_COMPLETED)`. For each finished future, settle it as today:
     - if a check failed and attempts remain, resubmit the repaired prompt at once.
       A retry is not checked for fit again: the repair margin is already inside
       `slot_ctx`, so the check on the original prompt covers it;
     - otherwise **yield `(key, outcome)`**, then pull one more item from `work` and
       submit it.
  4. Stop when `work` is exhausted and nothing is pending.
- **Fit check per prompt, just before submitting.** A prompt that cannot fit a slot is
  yielded at once as an outcome with the `fits_context` failure, as `handle` reports it
  today. It never reaches a slot, and the loop pulls the next item to fill its place.
- **Too-long prompts are found as they come, not up front.** `plan` knew the full list
  before the first generation; the stream knows it only at the end. Progress therefore
  reports a running `too_long` count, so a bad rate shows early. If the list is needed
  before the run, it comes from a token count at flatten time (open question 3), not
  from the stream.
- Outcomes come out in completion order, not input order. The key is what maps them
  back.
- `in_flight` comes from `[controller].in_flight` in `configs/qwen.toml`. The
  controller still never learns the slot count: `in_flight` is a budget, and the config
  comment says it should be about 2 × `n_parallel`.
- **There is one queue, and it is the model's.** A submitted prompt lives in the model's
  pool: `n_parallel` running (one per slot), the rest waiting in the pool's unbounded
  queue. The controller holds only a key → (future, prompt) record per outstanding
  prompt. It keeps the prompt because a retry is built from it. `in_flight` limits how
  many records the controller keeps, which is how it limits what it pulls from `work`.
- `finally:` cancels queued futures. Anything cancelled has no row, so resume re-runs
  it.
- **Whatever the consumer does between two `next()` calls runs on the loop thread.**
  The service's per-outcome work must be cheap: build the row, call `sink.add`. The
  queued headroom covers short pauses, not long ones.
- Progress: `_Meter` takes `total` when it is given. Without it, progress reports done
  count and throughput and omits the ETA.

`handle(prompts)` stays for small in-memory sets (tests, notebooks), rewritten as
`stream` over `enumerate(prompts)` that collects the outcomes back into input order.
There is then one loop, not two.

### `models/qwen_model.py` — split `plan` into `cache_shared_prefix` and `fits`

`plan(prompts)` does three things over the whole set: count tokens, set aside prompts
that are too long, and make every slot cache the start all prompts share. A stream never holds the
whole set, so these split:

```python
def cache_shared_prefix(self, sample: Sequence[Prompt]) -> None  # each slot caches the shared start once
def fits(self, prompt: Prompt) -> bool                           # one /tokenize call; size + max_tokens + margin <= slot_ctx
```

- `cache_shared_prefix` makes each slot process the start that every prompt shares
  (the system turn, about 1,600 of about 2,000 tokens) once and keep it cached, so
  every later prompt processes only its own ~400 tokens. It works out that start from
  the sample exactly as `plan` does now (common prefix, cut at the end of a turn), and
  skips the work if the slots already hold it. Every purpose/risk prompt shares the
  same system turn, so the first `in_flight` prompts are a sufficient sample. The name
  replaces `prime`, which did not say what it does.
- `fits` is one local HTTP call, a few milliseconds, so it is safe on the loop thread.
  The total is unchanged: `plan` already made one `/tokenize` call per prompt, all up
  front; `fits` makes the same N calls, one just before each submit.
- `/tokenize` does not wait behind generations. llama-server runs it on the HTTP thread
  that received it, using only the vocabulary: no slot, no GPU, no task queue. With
  `--threads-http` unset it starts at least `n_parallel + 2` HTTP threads, so one is
  free while all slots generate. This is from llama.cpp's server source and not yet
  measured here; acceptance item 6 checks it.
- `slot_ctx` is already fixed in config (5,120), not derived from the largest prompt.
  So a per-prompt fit check gives the same answer the whole-set plan gave.
- `plan` and its `Plan` result are removed. `handle` runs through `stream`, which
  calls `fits` and `cache_shared_prefix` itself, so nothing called `plan` any more.

`submit` does not change: its unbounded queue is where the headroom sits.

### The service

It owns the only loop a person reads:

```python
with store.Sink(run) as sink:
    todo = loader.pending(run, ...)
    work = ((lab.dsld_id, to_prompt(spec, lab)) for lab in todo)
    for dsld_id, outcome in controller.stream(work, checks, total=todo.total):
        sink.add(to_row(dsld_id, outcome))
```

The only per-label work is wrapping the loader's passage in the prompt spec
(`prompts.load(...)`) to make a `Prompt`. It happens inside the generator, one label at
a time, when the controller pulls. The poison guard (stop after too many consecutive failures) stays
in the service as workflow policy, and it checks each outcome as it arrives.

### `configs/qwen.toml`

```toml
[controller]
max_attempts = 3
# Prompts submitted but not yet settled. Must exceed [model].n_parallel so a freed
# slot always finds one queued; 2x is plenty, more only holds memory.
in_flight = 8
```

## Sizing

The queued headroom is `in_flight − n_parallel` = 4 prompts. The loop has to top the
queue back up before those 4 are taken. At the measured throughput (64 labels in 65.5s
on 4 slots, so about 1 label/s overall and about 4s per generation per slot), 4 queued
prompts cover several seconds. Against that:

| step on the loop thread | cost |
|---|---|
| read one label + build its `Prompt` | ms |
| `fits` (one `/tokenize`) | ms |
| settle + checks | sub-ms |
| `sink.add` | µs |
| `sink.flush` of 64 rows to local disk | ms |

None of these comes near the headroom. Nothing needs a background thread in v1.

## Failure and resume

- Only flushed rows exist. On a crash, what is lost is the unflushed buffer plus the
  prompts in flight. Resume re-runs those labels: it costs LLM time and never gives a
  wrong row.
- The cost of a crash is bounded by `flush_every` / `flush_seconds`. Pick them by how
  much re-work you are willing to accept. At about 1 label/s, 64 rows or 30s is well
  under a minute.
- A torn final line from a crash mid-flush is handled by the existing `read_rows` /
  `repair`.
- Ctrl-C: the controller's `finally` cancels queued futures, the sink's `__exit__`
  flushes what finished, and the server stops with the model's `close()`.

## Acceptance

1. **The model never waits on the pipeline.** Take the same 64 labels. Run the current
   `handle` on the full list (the ideal: everything queued) against `stream` over a lazy
   iterator. `stream` must take the same wall time, within run-to-run noise. Check that
   the GPU is free first, per the benchmark-isolation rule. Hand the benchmark to the
   user to run.
2. **Memory is bounded.** Peak RSS of a `stream` run over 1,000 labels is the same as
   over 64, give or take the id list.
3. **Resume is exact.** Kill a run mid-way. The resumed run's `pending.total` equals the
   original total minus the flushed rows. After completion, every id appears exactly
   once.
4. **Order independence.** `handle` via `stream` returns outcomes in input order and
   matches today's `handle` on the existing tests.
5. **Too-long prompts** are yielded as `fits_context` outcomes and never submitted.
6. **`/tokenize` is not blocked by generation.** Time `/tokenize` on a set of prompts
   against an idle server, then again while all `n_parallel` slots are generating. The
   busy times must stay in milliseconds. Hand the check to the user to run.

## Departures

From `docs/qwen-model-controller-design.md` and the agreed architecture:

- **The controller no longer hands the model the whole set.** It was agreed on
  2026-09-23 that the model measures the whole set in `plan`. With a stream it caches
  the shared prefix from a sample and checks fit per prompt. This holds because `slot_ctx` is now fixed
  config, not sized from the largest prompt.
- **Outcomes stream in completion order with a caller key**, not as an index-aligned
  list. The index-aligned list survives only as `handle`'s return value.
- **The controller no longer "divides a work set into batches".** There are no batches;
  there is a fixed in-flight budget.

## Questions and answers

From reviewing this spec on 2026-09-26.

**Do the model and the controller each keep a buffer of in-flight prompts?**
No, there is one queue and it is the model's. Submitted prompts sit in the model's
pool: `n_parallel` running, one per slot, and the rest waiting. The controller keeps
only a key → (future, prompt) record for each outstanding prompt. It keeps the prompt so
it can build a retry. `in_flight` limits how many records the controller holds.

**`plan` ran once over the whole set. Does the stream now have to run `fits` very often?**
It makes the same number of `/tokenize` calls: one per prompt. `plan` made them all up
front; `fits` makes each one just before its prompt is submitted. A call takes a few ms
against ~4 s per generation, so it does not slow the model. Retries are not checked
again, because the repair margin is already inside `slot_ctx`.

**`plan` found every too-long prompt at the start. The stream finds them one by one. Is
that a loss?**
The rows are the same: each too-long prompt gets a `fits_context` outcome and is never
submitted. What goes is knowing in advance. Progress shows a running `too_long` count to
make up for it. If the full list is needed before the run, compute token counts at
flatten time (open question 3).

**What does `prime` do?**
It makes each slot cache the start that every prompt shares (the system turn), so later
prompts skip those tokens. It is renamed `cache_shared_prefix` and runs once per run.

**`/tokenize` is llama-server's work too. Does it wait while the slots are generating?**
No. llama-server runs it on the HTTP thread that received the request, using only the
vocabulary: no slot, no GPU, no task queue. At least `n_parallel + 2` HTTP threads exist,
so one is free. This comes from llama.cpp's source and has not been measured here yet;
acceptance item 6 checks it.

**Who asks for more work: the model asks the controller, and the controller asks the
service?**
Nobody asks. The controller's loop is the only active part. The model runs whatever is
in its queue. The service and loader are a lazy generator whose code runs only when the
controller calls `next(work)`. The controller keeps `in_flight` prompts outstanding. A
slot taking a queued prompt and a future completing are the same event, so each
completed future tells the controller one queued prompt was just taken. It pulls one
label, checks that it fits and submits it. The queue is one short for a few ms, which
the headroom covers.

**If a prompt does not fit, does something notify the controller so it sends another?**
No notification is needed. The controller calls `fits` itself and waits for the answer
before submitting. On `False` it yields the `fits_context` outcome right away. The
prompt was never submitted, so the outstanding count has not risen, and the refill loop
simply pulls the next label:

```python
while outstanding < in_flight and (item := next(work, None)) is not None:
    key, prompt = item
    if not model.fits(prompt):
        yield key, too_long_outcome(prompt)   # never submitted
        continue                              # pull the next one
    futures[model.submit(prompt)] = (key, prompt)
```

The model never sees a too-long prompt.

## Open questions

1. Where does the sink's config (`flush_every`, `flush_seconds`) live? It is a
   repository setting, not a Qwen one, so it does not belong in `qwen.toml`.
   Constructor defaults are the proposal for now.
2. Does `pending` replace `load(resume_from=...)`, or sit beside it? It overlaps with
   `load` except for laziness.
3. The flattened file for the full corpus: `load_flattened` defaults to the pilot
   passages. The full run needs the whole DSLD cleaned and flattened into the same
   format first. Is that a one-off job run ahead of time, or part of this run? If it is
   run ahead of time, it could also store each label's token count, which gives the
   full too-long list before the GPU starts. The count depends only on the tokenizer
   and the prompt spec, not on model state.
