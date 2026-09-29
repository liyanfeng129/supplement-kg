# Purpose/risk service: coordinate loader, controller and store

- **Status:** proposed
- **Date:** 2026-09-29
- **Scope:** new `services/purpose_risk/service.py`, new `engine/notify.py`, new
  `configs/engine.toml`, new `repositories/prompts/purpose_risk/parse.py`, and changes
  to `controllers/qwen_controller.py`, `models/qwen_model.py`,
  `repositories/prompts/schema.py` and `repositories/dsld/store.py`.

Builds on `docs/specs/0002-streaming-run.md`, which built the loader's `pending`, the
store's `Sink` and the controller's `stream`. This spec adds the service that joins
them, and amends 0002 where listed under [Departures](#departures).

## Scope of the service

The service **only coordinates**: loader → controller → store, for purpose/risk.
Boilerplate filtering, grounding and scoring are future work that will be added to the
service later; none of it is in this build.

## Flow

```
entry point (notebook / pipeline script)          configures logging
        │  with PurposeRiskService() as svc: svc.run(run=None | path, ids, limit, …)
        ▼
PurposeRiskService
  1. prompts.load("purpose_risk") ─► PromptSpec(system, checks, parse, stop, max_tokens)
  2. manifest: new run → write it │ resume → read selection, refuse if config_hash differs
  3. loader.pending(run, …) ─► Pending (lazy labels, total, n_already_done)
  4. store.Sink(run) as sink
  5. for key, outcome in controller.stream(work, spec.checks, spec.parse, total, done):
         row = to_row(key, outcome) ─► sink.add(row)
         count failures ─► notify() once if over threshold
  6. end: notify(summary) + one print │ ModelUnavailable: flush, notify, re-raise
        │ pulls lazily                      │ yields (dsld_id, Outcome)
        ▼                                   ▲
QwenController ── creates & owns ──► QwenModel ──► llama-server
```

## Responsibilities

### Entry point (`pipeline/…` or a notebook)

Calls `logging.basicConfig(level=INFO)` once, opens the service, calls `run(...)`.
Never builds a model or controller, never touches loader or store.

### `services/purpose_risk/service.py` — `PurposeRiskService`

```python
class PurposeRiskService:
    def __init__(self, controller: QwenController | None = None): ...
    def __enter__(self) -> "PurposeRiskService": ...
    def __exit__(self, *exc) -> None: ...          # controller.close()
    def run(self, run=None, path=None, ids=None, limit=None, order=None, seed=None) -> None
```

- Declares its own collaborators: it is for purpose/risk only. It creates the
  controller (which creates the model); it never creates a model itself. `controller`
  is an optional seam so tests pass one built on a fake model.
- `run=None` starts a new timestamped run directory; `run=<path>` resumes it. On
  resume, the selection arguments are read from the manifest, not passed again.
- Loads the prompt spec, writes or checks the manifest, and owns the one loop:
  label → `Prompt` (lazy) → `controller.stream` → row dict → `sink.add`.
- **Failure policy:** a label whose outcome is not ok is written as a failed row and
  the run continues. The service counts failed labels; once more than
  `[notify].fail_rate` of labels have failed after at least `[notify].fail_min_labels`,
  it calls `notify` **once**. The run does not stop.
- Calls `notify` at the end with the counts per status, and on `ModelUnavailable`
  (then re-raises; the sink has flushed).
- Returns nothing. Prints one summary line at the end. Does no progress, no per-label
  logging, no parsing, no checking, no retrying, and knows nothing about hosting.

### `controllers/qwen_controller.py` — `QwenController`

- **Creates and owns the model:** `QwenController(model=None)` builds `QwenModel()`
  when none is passed; `close()` closes it. The service controls the model only
  through the controller.
- As today: readies the model, caches the shared prefix, keeps `in_flight` prompts
  outstanding, checks fit before submit (too long → `too_long` outcome, never sent),
  runs checks and repairs failed answers up to `max_attempts`.
- **New — per-request errors are retried** within `max_attempts` like a failed check.
  After the last attempt the label is yielded as an `error` outcome, and the stream
  continues.
- **New — parses.** `stream(..., parse=...)` calls the prompt's parser on `ok`
  answers only and puts the result in `Outcome.parsed`; failed outcomes have
  `parsed = None`. The controller runs the parser but does not own the format.
- **Progress and ETA** stay here (`_Meter`, `_report`):
  - `stream(..., n_already_done=...)` so a resumed run reports
    `40001/88000 (this session 1/48000)`;
  - rate and ETA over a rolling window of the last ~200 labels, not the whole-run
    average, so a slowdown shows within minutes.
- Lets `ModelUnavailable` through: the only thing that stops a run.

### `models/qwen_model.py` — `QwenModel`

- **New:** when a request fails, the model checks its own health. Server alive → the
  request's error is raised as a per-request error (the controller retries it). Server
  gone (process exited, connection refused, health check fails, CUDA/OOM) →
  `ModelUnavailable`. No automatic restart in this build.

### `repositories/prompts/purpose_risk/` — the prompt and its format

- `checks.py` (exists) makes the LLM's output follow the short line format that is
  cheap to generate.
- **New `parse.py`** turns that format into JSON fields:
  `P: supports restful sleep | 4` → `{"text": "supports restful sleep", "score": 4}`
  in `purpose`, `R:` lines in `risk`.
- `PromptSpec` gains `parse`; `prompts.load` fills it from `parse.py`. The line format
  lives only in this folder.

### `repositories/dsld/loader.py` — `pending` (built in 0002)

Ids up front minus done ids; labels read lazily; supplies `total` and
`n_already_done`.

### `repositories/dsld/store.py` — `Sink` (built in 0002)

Buffers dict rows and appends at `flush_every` rows or `flush_seconds`; repairs a torn
trailing line on open; flushes on exit, including on an exception.
**New:** flush settings are read from `configs/engine.toml` `[store]`, constructor
arguments still win. The store takes plain dicts and never imports `Outcome`.

### `engine/notify.py` — new

```python
def notify(subject: str, body: str) -> None
```

Always logs. Also emails through Gmail SMTP (`smtplib`, app password) when the
credentials are set in `.env`; without them it only logs. At most three emails per
run: the failure-rate warning, run finished, model unavailable.

### `configs/engine.toml` — new, global

```toml
[store]
flush_every = 64
flush_seconds = 30.0

[notify]
fail_rate = 0.05       # warn when failed labels exceed this share...
fail_min_labels = 200  # ...once at least this many are done
```

Mail address, app password and recipient live in `.env`, never in a committed file.

## Interaction modes

| from → to | mode |
|---|---|
| service → loader | lazy pull: a label is read only when the controller calls `next(work)` |
| service → controller | one blocking generator, `stream()`; the controller's loop is the only active part |
| controller → model | `submit` returns a future; `wait(FIRST_COMPLETED)` refills a freed slot at once |
| service → store | `sink.add(dict)`, µs on the loop thread; flushes are small synchronous appends |
| service → notify | direct call, at most three times per run |

Dependency direction: entry → service → controller → model; the service also uses
the prompts, loader and store repositories. Nothing calls up.

## Rows

One row per label in `data/dsld/runs/<ts>/purpose_risk.jsonl`:

```json
{"dsld_id": 123, "status": "ok|failed_checks|too_long|error",
 "purpose": [{"text": "...", "score": 4}], "risk": [{"text": "...", "score": 3}],
 "text": "<raw answer>", "attempts": 1, "failed": [],
 "prompt_tokens": 1987, "completion_tokens": 140, "latency_ms": 4100,
 "config_hash": "..."}
```

- Every label gets a row, failed ones included, so resume skips it. Failed rows have
  empty `purpose` / `risk` and keep the raw `text` for inspection.
- The raw `text` stays so a better parser can re-derive the terms without the GPU.
- Re-running failed labels is not built. When needed: a new run directory seeded with
  the old run's non-ok ids.

## Manifest and resume

- A new run writes the manifest: selection arguments and `config_hash`.
- `config_hash` covers the prompt spec (system, stop, max_tokens, check names), the
  model, the `[sampling]` settings and `max_attempts`. Not `in_flight`, `n_parallel`
  or the flush settings: they change speed, not answers.
- Resume reads the selection from the manifest and refuses when `config_hash`
  differs. The fix is a new run directory.
- After a crash only the unflushed buffer and the prompts in flight are re-run.

## Build order

1. `parse.py` and `PromptSpec.parse`.
2. Model: per-request error vs `ModelUnavailable`.
3. Controller: `model=None` + `close()`; error retry and `error` outcome; `parse`;
   `n_already_done` and rolling ETA.
4. `configs/engine.toml` and the sink reading it.
5. `notify.py`.
6. The service, tested end to end on a fake model in a `tmp_path` run directory.
7. A notebook running it on the GPU, handed to the user to run.

## Departures

From spec 0002 and the agreed architecture:

- **The controller creates the model.** Before, the caller built `QwenModel` and passed
  it in. The model still owns its lifecycle (ready, health, close); the controller now
  owns the model's existence, and the service reaches the model only through it.
- **The controller parses.** It runs a parser supplied with the prompt spec, as it
  already runs the checks. The format stays in `repositories/prompts/`.
- **A generation error no longer ends the stream.** It is retried and then yielded as
  an `error` outcome; only `ModelUnavailable` ends the stream.
- **The poison guard is a warning, not a stop.** 0002 said "stop after too many
  consecutive failures"; the service now keeps going and notifies once.
- **Sink config** (0002 open question 1) lives in `configs/engine.toml`.

## Acceptance

1. On a fake model, a service run over N labels writes N rows, one per id, with
   statuses matching the fake's outcomes.
2. Killing a run and resuming it writes every id exactly once; a resume with a changed
   prompt is refused.
3. A fake per-request error is retried, then written as `error`; the run continues.
4. A fake `ModelUnavailable` stops the run with every finished row flushed.
5. Crossing the failure threshold calls `notify` exactly once.
6. `ok` rows carry parsed `purpose` / `risk`; failed rows carry empty lists and the
   raw text.
