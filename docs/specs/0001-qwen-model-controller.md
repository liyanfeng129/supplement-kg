# Qwen model and controller

- **Status:** built; describes the code as of 2026-09-29
- **Date:** 2026-09-28, amended 2026-09-29
- **Scope:** `engine/models/qwen_model.py`, `engine/controllers/qwen_controller.py`,
  and `configs/qwen.toml`, which both read.

Governed by ADR 0002; the places this differs from it are listed under
[Departures](#departures). The streaming run that shaped the controller is
specified in `docs/specs/0002-streaming-run.md`. The purpose/risk service that
drives it, and the changes that service needed here (the controller owns the model,
runs the parser, retries failed requests), are in
`docs/specs/0003-purpose-risk-service.md`.

## The rule

**The local LLM is the bottleneck. Its queue must never be empty while work
remains.** Everything else — reading a label, building a prompt, checking an
answer, saving a row — takes milliseconds against seconds per generation, and
exists to keep the model fed.

## Who owns what

| | Model (`QwenModel`) | Controller (`QwenController`) |
|---|---|---|
| **Owns** | Execution: the server, the slots, the queue; its own lifecycle and health | The model's existence; the in-flight budget, validation, parsing, retry, progress |
| **Knows** | `n_parallel`, `slot_ctx`, its tokenizer, its chat format | `in_flight`, `max_attempts`, the checks by name, that a parser exists |
| **Never knows** | What a prompt is for; who is calling | The slot count; that a server exists; what a label is; the answer format |

The controller imports `QwenModel` concretely: one controller per model. A
future `claude_controller` pairs with `claude_model` the same way. Neither
imports anything from `services` or `repositories`.

**The controller creates the model.** `QwenController()` builds `QwenModel()`;
`close()` closes it. A caller — the service — reaches the model only through the
controller and never builds one. Passing a model in (`QwenController(model)`) is
the seam tests use. The model still owns its lifecycle (ready, health, close);
the controller owns whether it exists.

## The model

**One per process.** `QwenModel(...)` always returns the same object; there is
no `__init__`, because Python would re-run it on every call and reset the
shared model. A setting left as `None` keeps its current value. A different
setting is refused while prompts are queued or running, restarts the server
when idle, and is simply taken if the server was never started. `close()` lets
go of the model; the next `QwenModel()` builds a fresh one.

**Hosted on `llama-server`, privately.** `ready()` starts it on a free local
port, the model talks to it over HTTP, `close()` stops it, and the server dies
with its Python process. Nothing above the model knows a server exists.

**Six public verbs:**

| verb | what it does |
|---|---|
| `ready()` | Idempotent. Refuses unless the card has room, the weights actually reached the GPU, and the model answers "Hello" in coherent language. Then starts the pool. Refuses — does not restart — a server it started that has since exited. |
| `generate(prompt)` | One blocking HTTP call; returns text, token counts, latency, stop reason. A failed request raises either its own error or `ModelUnavailable` (see [Failed requests](#failed-requests)). |
| `fits(prompt)` | One `/tokenize` call: does prompt + output budget + repair margin fit one slot? |
| `cache_shared_prefix(sample)` | Every slot reads the start all prompts share (the ~1,600-token system turn) once and keeps it cached. Worked out from a sample; skipped if already cached. |
| `submit(prompt)` | Queues one prompt, returns a `Future[Generation]` at once. Never blocks, never refuses. |
| `close()` | Cancels queued prompts, stops the server, frees the card. Idempotent. |

**GPU-only, and it checks.** A CPU fallback gives correct rows at ~100x the
cost with no error. So `ready()` checks free VRAM before starting, measures the
VRAM the load took against the weights' size, and runs a deliberately weak
coherence check on a "Hello" reply — enough to catch a runtime emitting fluent
garbage, not a subtly degraded model. Any failure raises `ModelUnavailable`
naming whatever process holds the card.

**Constrained output** is asked for by passing a schema on the `Prompt`; the
server compiles it to a grammar, at about 4x slower decode. `schema=None` is
free sampling.

### Failed requests

When a `/completion` request fails, the model checks its own health before
deciding whose failure it is:

| the server | the model raises | what happens above |
|---|---|---|
| process exited, or already closed | `ModelUnavailable` | the controller lets it through; the run stops |
| reports a CUDA or out-of-memory failure | `ModelUnavailable` | as above |
| alive, but `/health` does not answer ok (5 s timeout) | `ModelUnavailable` | as above |
| alive and healthy | the request's own error, as it was | the controller retries it |

**No automatic restart in this build.** Whatever killed the server — a CUDA
fault, the card taken — would likely kill the next one, so a person decides.
`ready()` therefore refuses a server this model started that has since exited,
rather than starting a new one; without that, the next queued prompt's
`generate()` would restart it silently from a pool thread. `close()` and then
`ready()` start a fresh server.

The health check covers `/completion` only. A failure of `/tokenize` (`fits`) or
of the prefix cache on a server that is alive but hung is raised as it is, not as
`ModelUnavailable`; an exited server is caught there too, by `ready()`.

### Slots and the queue

`n_parallel = 4` (`[model]`) is used in exactly three places, so they always
agree:

- the server is started with 4 slots (`-np 4`);
- its context is 4 × `slot_ctx`, split evenly (`--no-kv-unified`), so every
  slot holds exactly 5,120 tokens and `fits` is exact;
- the thread pool has 4 workers.

`submit` puts a `generate` call into that pool. If a worker is free it runs
at once; otherwise it waits in the pool's first-in, first-out queue. **The
queue has no size limit**: the model would accept any number of prompts. When
a slot finishes, its worker takes the next queued prompt at once, inside the
model, without the controller doing anything. The server never sees more than
4 requests at a time; the rest wait in this process, where `close()` can
cancel them.

A `Future` is pending while queued or running, and done once `generate`
returns or raises. The model's own `_in_flight` count (queued + running)
limits nothing; it exists only to refuse a settings change mid-run.

## The controller

**Construction.** `QwenController(model=None, max_attempts=None, in_flight=None)`
builds `QwenModel()` when no model is given; the budget and in-flight count come
from `[controller]` unless passed. `close()` closes the model.

**Two entry points:**

- `stream(work, checks, parse, total, n_already_done, on_progress)` takes a lazy
  iterable of `(key, Prompt)` and yields `(key, Outcome)` as each settles, in
  completion order. The key is opaque; the caller uses it to map outcomes back.
  `total` is the prompts in this `work`; `n_already_done` is what earlier sessions
  of a resumed run finished, counted into progress.
- `handle(prompts, checks, on_progress)` is `stream` over an in-memory list,
  collected back into input order. For tests, notebooks and small sets. There
  is no async twin: parallelism is the model's, so an async caller runs
  `handle` on a worker thread.

**`Outcome`** is `generation` (the last one), `ok`, `attempts`, `failed` (check
names from the final attempt, or `fits_context` / `error`) and `parsed` (the
parser's result on `ok` outcomes, else `None`).

**`in_flight = 8`** (`[controller]`) is the number of prompts submitted but not
yet settled. It must exceed `n_parallel`; about 2x is enough, and more only
holds memory. The controller never reads `n_parallel`, so the ratio is kept by
hand, through the comment in `qwen.toml`.

### Laziness goes one way

```
loader ──lazy──► controller ──eager, up to 8──► model (4 running + 4 queued)
```

A label is read and built into a prompt only when the controller calls
`next(work)`, and it asks only while fewer than 8 are outstanding. Label 9
does not exist in memory until one of the first 8 has settled. But a pulled
prompt is submitted at once — the controller never holds one back waiting for a
slot. So `in_flight` bounds how many labels exist at once; it is not a waiting
room inside the controller. All 8 are with the model, 4 running and 4 queued.

The 4 queued are the **headroom**. A freed slot takes one at once, and the
controller refills behind it later. Without them, every freed slot would idle
while the controller read a label, built a prompt, tokenised it and submitted
it.

### The loop, step by step

1. **Peek.** Take the first `in_flight` items from `work`. If there are none,
   return without ever starting the server. Otherwise ready the model, cache
   the shared prefix from those prompts, and put them back in front of `work`
   so none is skipped.
2. **Refill.** While fewer than `in_flight` are outstanding and `work` is not
   used up: pull one item, ask `fits`, and submit it. A prompt that does not
   fit is yielded at once as a `fits_context` outcome, is never submitted, and
   does not count as outstanding — so the next label is pulled in its place.
   The refill ends when 8 are outstanding or `work` runs out.
3. **Finish** if nothing is outstanding (work used up, all settled).
4. **Wait** until at least one outstanding future is done. This blocks only if
   none is done yet; if one already is, it returns at once.
5. **Settle one** — the oldest done future, in submit order — then go back to
   step 2.

The controller keeps one record per outstanding prompt: future → (key,
original prompt, attempts finished so far). The original prompt is kept so a
retry can be built from it.

**One at a time, on one thread.** The controller settles, yields and refills
one prompt per pass. If several futures are done at once — two slots finishing
together, or the caller slow to ask for the next outcome — the rest stay in the
record and are settled on the following passes, each of which returns from the
wait at once. Nothing is lost, and a refill runs between every settle, which
keeps the model's queue topped up sooner than settling a whole batch first.

**Fast enough not to matter.** A send is a few milliseconds: read the label,
build the prompt, one `/tokenize`, queue it. A generation is about 4 s. At the
start the refill sends 8 back to back and every slot is busy almost at once;
in steady state one settle leads to one send, long before a slot needs it; after
a pause, the next refill sends several in a row. The controller spends most of
its time blocked in the wait.

### Settling a result

Each settle counts as one attempt. First the future's result is taken; a request
that failed is handled before any check runs. Otherwise the checks run against
the generation, with the original prompt (so a rule about the input sees the
input, not a repaired copy). Then:

| result | what happens |
|---|---|
| `ModelUnavailable` (or the future was cancelled) | propagates; the stream ends |
| request failed, attempts left | resubmit the original prompt as it was; no yield, no new label pulled |
| request failed, no attempts left | log a warning **and** yield `ok=False`, `failed=["error"]`, empty text, `stop_reason="error"` |
| passed | yield `ok=True`, with `parsed = parse(text)` when a parser was given |
| failed a check, attempts left | resubmit a repaired copy; no yield, no new label pulled |
| failed a check, no attempts left | log a warning **and** yield `ok=False` with the failing check names; `parsed=None` |

Every submitted label, and every too-long one, ends in exactly one yield —
unless `ModelUnavailable` ends the stream first. A permanent failure still gets a
row, so resume does not retry it forever, and the service's failure count can
see it.

**The parser comes with the prompt**, like the checks (`PromptSpec.parse`, from
`repositories/prompts/<id>/parse.py`). The controller runs it on passing answers
only, without knowing the format. A parser that raises is a bug and propagates,
as a check that raises does.

### Retries

- **A retry is a new submission.** It gets a new future; the old one is
  removed. The record carries over the same key, the original prompt and the
  attempt count, which continues from where it was.
- **Failed checks and failed requests share one budget**, `max_attempts`.
- **A check repair is built from the original prompt**, with each failed check's
  message appended to the user turn. Notes do not pile up across attempts, and
  the prompt cannot grow past what `fits` approved plus the repair margin
  reserved inside `slot_ctx` — which is why a retry is not checked for fit
  again.
- **A failed request is re-sent as it was.** No answer came back, so there is no
  rule to cite. This also drops any repair note an earlier attempt had earned.
- **It goes to the back.** The new record is the newest, and the new prompt
  joins the end of the model's queue. The outstanding count does not change,
  so the retry uses the room an extra label would otherwise take.
- **It cannot slip to the end of the run.** At most 7 prompts are ever ahead of
  it, and every label pulled afterwards queues behind it. It falls roughly one
  `in_flight` window back per retry; not a strict bound, since generation
  lengths vary and a label can retry up to `max_attempts − 1` times. Nothing
  downstream depends on order.

### Handing back

At a `yield`, the controller pauses and the caller gets `(key, outcome)`.
Whatever the caller does before asking for the next one runs on the
controller's thread, while the model keeps working through its queue. It must
be cheap — build a row, append it to a buffer. The 4 queued prompts cover
short pauses, not long ones.

### Stopping

The cleanup runs once, when `stream` ends for any reason: work finished, the
caller breaks out, `ModelUnavailable`, another exception (in the controller or
raised by the caller while paused at a yield), or Ctrl-C. It cancels every
outstanding future. Ones still queued are removed; ones already running on a
slot finish, or stop with `model.close()`. A cancelled prompt has no outcome, so
a resumed run re-runs it. After a normal finish nothing is outstanding and the
cleanup does nothing.

### Progress

The model reports tokens and latency per generation. The controller counts
settled units against the wall clock — not summed latency, because four
concurrent 6 s generations take 6 s, not 24 — and reports done/total,
throughput, a running `too_long` count, and an ETA when `total` is given. The
clock starts after the model is ready, so the load never inflates the ETA.

- **Rolling, not whole-run.** Throughput and the ETA are over the last 200 units
  (`_ETA_WINDOW`), so a slowdown shows within minutes rather than being averaged
  away over days.
- **Resumed runs count the whole run.** With `n_already_done`, done/total include
  what earlier sessions finished, and this session's share is reported beside it:
  `40001/88000 (this session 1/48000) · …`.

### Errors

`ModelUnavailable` from `ready()` is re-raised as "none of the N prompts were
run, because…". `ModelUnavailable` from a request is let through unchanged: it
is the only thing that stops a run. A request's own error is retried, then
yielded as `error`. A check or parser that raises is a bug and propagates.

## Checks

A `Check` (`models/base_model.py`) is a name, a `passes(prompt, generation)`
test, and a message. Checks are atomic, one rule each, and live with their
prompt (`repositories/prompts/<id>/checks.py`). The message is written to the
model and appended on a retry; the name is for logs and `Outcome.failed`, and
the model never sees it. The controller runs checks without understanding
them.

## Settings

`configs/qwen.toml` is committed, not `.env`: these are experiment inputs, and
belong in git, diffable between runs. Read through `engine/config.py`
(`read_toml`, `QWEN_CONFIG_PATH`), which returns `{}` for a missing or
malformed file rather than raising.

| table | read by | holds |
|---|---|---|
| `[model]` | model | paths, `n_parallel`, `slot_ctx`, `ubatch`, speculative decoding, `min_vram_mib` |
| `[sampling]` | model | seed, temperature, top-p/k, `max_tokens`, `enable_thinking` |
| `[controller]` | controller | `max_attempts`, `in_flight` |

Precedence everywhere: an explicit constructor argument, then the file, then
a hardcoded fallback. A value of the wrong type (including a boolean where an
integer is wanted) falls back like a missing one.

Engine-wide settings that are neither the model's nor the controller's — the
store's flush thresholds and the failure-warning thresholds — live in
`configs/engine.toml` (spec 0003).

## Responsibilities

| # | Responsibility | Model | Controller |
|---|---|---|---|
| 1 | Generation | Prompt in; text, token counts, latency, stop reason out | — |
| 2 | Prompt format | ChatML, thinking switched off | — |
| 3 | Constrained output | Passes a schema to the server as its grammar | — |
| 4 | Lifecycle | Starts and stops `llama-server`, frees the GPU; never restarts a dead one | Creates the model; readies it only once there is work; `close()` closes it |
| 5 | Readiness checks | VRAM free, weights on the GPU, a coherent "Hello" | — |
| 6 | Health | After a failed request: process alive, no CUDA/OOM, `/health` ok | — |
| 7 | Hosting and tuning | 4 slots, speculative decoding, `-ub 2048` | — |
| 8 | Concurrency | At most 4 running; a freed slot takes the next queued prompt | Keeps `in_flight` outstanding; pulls one more as each settles |
| 9 | Measuring the work | `fits` per prompt; `cache_shared_prefix` from a sample | Caches the prefix once from the first prompts; asks `fits` before each first submit |
| 10 | Settings | `[model]`, `[sampling]` | `[controller]` |
| 11 | Timeouts | 15 min to start, 30 min per request, 5 s health check after a failure | — |
| 12 | Validation | — | Runs the checks |
| 13 | Parsing | — | Runs the prompt's parser on `ok` answers |
| 14 | Retries | — | Up to `max_attempts`, for failed checks and failed requests alike |
| 15 | Repair | — | Appends each failed check's message to the original prompt |
| 16 | Giving up | — | `ok=False` with the reasons, or `error`; the run carries on |
| 17 | Error handling | `ModelUnavailable` naming what holds the GPU, or why the server is gone | "None of the N prompts ran, because…"; lets `ModelUnavailable` through |
| 18 | Progress | Tokens and latency per answer | Done/total (whole run on resume), rolling throughput and ETA, too-long count |
| 19 | Result shape | One `Generation` per prompt | `stream`: `(key, outcome)` in completion order. `handle`: input order |
| 20 | Cleanup | `close()` cancels queued work; the server dies with its process | Cancels what it submitted when `stream` ends |
| 21 | Logging | The server's log file; a warning when a settings change restarts the server | Misfits, retries, give-ups, progress |
| 22 | One instance | `QwenModel()` always returns the one model | Builds it with `QwenModel()` |

## Testing

Tests call only the public verbs. The model's tests fake `nvidia-smi`, the
server process and its HTTP API underneath; a failed request is a scripted
`/completion` error, a dead server a fake process with an exit code, and a
hung one a `/health` that refuses the connection. The controller's tests use a
scripted `BaseModel`, since the singleton cannot be subclassed per test; a
scripted reply that is an exception is a failed request.

## Measurements behind the code

Run ids are under `data/benchmarks/`; apparatus `benchmarks/parallel_slots.py`.

| decision | measured | run |
|---|---|---|
| `llama-server`, not in-process `llama_cpp` | 32 labels: 99.5s in-process vs 74.6s one server slot | `parallel-20260923T145320Z` |
| 4 slots | 32 labels: 1 slot 74.6s, 2 57.1s, **4 53.7s**, 6 55.6s, 8 56.2s. Prompt reading already runs at full width, so more slots only queue | `parallel-20260923T145320Z` |
| refill a freed slot at once, no batch barrier | 73.5s batched vs 73.9s one at a time vs 53.7s continuous pool | `parallel-20260923T152816Z` |
| MTP speculative decoding, 3 drafted tokens | 64 labels: 101–105s defaults → 77s. 5 drafted tokens 2.5% faster (noise) for 1.2 GB more VRAM | `parallel-20260923T154114Z`, `-155732Z` |
| `-ub 2048` + cached shared prefix | 77s → 65.5s; prompt tokens read per label 1,460 → 444. Without the cached prefix, 2,048 was slower (129s) | `parallel-20260923T154114Z`, `-155732Z` |
| `--no-kv-unified` | `--kv-unified` changed nothing; per-slot context makes `fits` exact | `parallel-20260923T154114Z`, `-155732Z` |
| input order, not longest-first | longest-first read 32,371 prompt tokens vs 25,317: its first requests missed the cached system turn | — |
| `min_vram_mib = 21500` | peak ~20.9 GB at 4 slots × 5,120 tokens with the settings above | — |
| schema costs ~4x decode | grammar sampler vs free sampling | — |

Still to measure (spec 0002, acceptance): that `stream` over a lazy iterator
matches `handle` over a full list in wall time, and that `/tokenize` stays in
milliseconds while all slots generate.

## Open

1. **Hung-server detection outside `/completion`.** `fits` and
   `cache_shared_prefix` do not run the health check, so a server that is alive
   but hung surfaces there as a plain network error after the request timeout,
   and the service's "flush, notify, re-raise" path is skipped. Covering them
   means one shared health check, which is a new function to agree first.
2. **A failed request drops an earlier attempt.** If attempt 1 failed a check and
   the last attempt's request failed, the outcome is `error` with empty text; the
   earlier answer and its check names are lost.

## Departures

From ADR 0002 and ADR 0001, still to be amended there:

1. **Progress is split across both layers.** The model reports tokens and
   latency per generation; the controller turns them into done-of-total and
   an ETA.
2. **The model verifies itself.** The "Hello" check sits inside `ready()`: a
   model that can load and emit garbage has not finished becoming ready. After
   a failed request it also judges its own health.
3. **Validation with retry is the controller's**, driven by checks that come
   with the prompt, because the controller owns the loop.
4. **`llama-server` is used.** ADR 0001 says it is not; superseded by the
   measurements above.
5. **The controller does not divide work into batches.** It keeps a fixed
   in-flight budget and refills one at a time; the model never sees the whole
   work set.
6. **The controller creates the model** (spec 0003). Before, the caller built
   `QwenModel` and passed it in; the controller was "handed that model".
7. **The controller parses** (spec 0003), with a parser supplied by the prompt;
   the format stays in `repositories/prompts/`.
8. **A generation error no longer ends the stream** (spec 0003). It is retried
   and then yielded as `error`; only `ModelUnavailable` ends the stream.
