# Qwen model and controller: the build

- **Status:** built; amended 2026-09-23 for hosting and concurrency (see the end)
- **Date:** 2026-09-23
- **Scope:** `engine/models/qwen_model.py` and `engine/controllers/qwen_controller.py`,
  plus the config file they read. Settled by interview; this is the record to build from.

Governed by ADR 0002, which it amends in three places — see [Departures from ADR
0002](#departures-from-adr-0002).

## Files

```
configs/
└── qwen.toml                    NEW   committed hyperparameters

engine/
├── models/
│   ├── base_model.py            EXISTS, unchanged
│   └── qwen_model.py            EXISTS, two changes
└── controllers/
    └── qwen_controller.py       NEW
```

Three files touched, one new. `engine/controllers/__init__.py` already exists and
stays empty.

## `configs/qwen.toml` — new

```toml
[model]
model_path = "..."      # the .gguf
n_ctx = 8192
min_vram_mib = 18000

[sampling]
seed = 0
temperature = 0.0
top_p = 1.0
top_k = 0
max_tokens = 2000
enable_thinking = false

[controller]
max_attempts = 3
```

Two model tables because they answer different questions — what is loaded versus how
it samples — and one controller table so the retry budget is a committed setting too.

**Committed, not `.env`.** `engine/config.py` is `.env`-backed and gitignored: right
for API keys, wrong for hyperparameters. Model settings are experiment inputs — they
belong in git, diffable between runs, and quotable in a manifest so a result can be
traced to the settings that produced it. Read with stdlib `tomllib`; no new dependency.

Named per model, so a second model gets its own file and the one-controller-one-model
symmetry holds.

## `engine/models/qwen_model.py` — two changes

Everything currently in the file stays: `ready`, `generate`, `close`,
`_check_gpu_offload`, `_render`, `_grammar_for`, `gpu_holders`, `free_vram_mib`,
`_unavailable_message`. The controller depends on all of it.

### 1. Defaults from the config file

```python
DEFAULTS_PATH = REPO_ROOT / "configs" / "qwen.toml"

def load_defaults(path: Path = DEFAULTS_PATH) -> dict
    """Flatten the TOML into constructor kwargs. Missing file -> hardcoded defaults."""
```

`__init__` changes from eleven literal defaults to `None` sentinels resolved against
`load_defaults()`. An explicit argument still wins. Construction stays free — a missing
or malformed file must not raise at import, so the hardcoded fallbacks remain.

### 2. A hello check inside the lifecycle

```python
HELLO_PROMPT = Prompt(system=..., user="Hello")   # module constant

def _check_alive(self) -> None
    """Generate a greeting; raise ModelUnavailable if it isn't coherent language."""
```

Called at the end of `ready()`, after the weights load. Readiness therefore means
*loaded and generating sense*, not merely loaded. The idempotence guard means it runs
once per load, not per batch. Cost is one short generation on top of a ~40s load.

**Why it exists.** This model's CUDA path can load normally and emit fluent garbage —
no error, plausible output, only the clock and the content to tell you. A model that
reports ready can be lying, so the lifecycle verifies itself.

**Its limit, stated honestly.** The coherence test is weak: non-empty, plausible
length, actual words. It catches a runtime emitting garbage. It will not catch a model
that is subtly degraded but still answers "hello" fine. That is the accepted cost of
keeping the model free of anything workflow-shaped — a stronger check would need
domain fixtures, and those belong above.

## `engine/controllers/qwen_controller.py` — new

### Shapes

```python
@dataclass(frozen=True)
class Attempt:
    """One try at one prompt: what came back, and which checks it failed."""
    generation: Generation
    failed: list[str]           # check names, empty == passed

@dataclass(frozen=True)
class Outcome:
    """The end state of one unit of work, however many attempts it took."""
    generation: Generation      # the last one, passing or not
    ok: bool
    attempts: int
    failed: list[str]           # names from the final attempt
```

`Outcome` rather than a bare `Generation` because giving up must state *why*: that
needs somewhere to put the failing check names and the attempt count. Index-aligned
with the input, so the service maps results back to labels by position.

### Surface

```python
class QwenController:
    def __init__(self, model: QwenModel, max_attempts: int | None = None)
        """Holds one model. Budget from configs/qwen.toml unless overridden."""

    def handle(self, prompts, checks=(), on_progress=None) -> list[Outcome]
        """Run every prompt to an outcome, sequentially. Ready the model first."""

    async def handle_async(self, prompts, checks=(), on_progress=None) -> list[Outcome]
        """Same contract; execution strategy is this layer's and the model's business."""

    def _run_one(self, prompt, checks) -> Outcome
        """Generate, check, repair, re-send, up to the budget."""

    def _repair(self, prompt: Prompt, failed: list[str]) -> Prompt
        """Append which named checks failed to the user turn."""

    def _ready(self) -> None
        """model.ready(), translating ModelUnavailable into a readable failure."""
```

### The retry loop

`_run_one` generates, runs the checks, and on failure composes a repair and re-sends,
up to 3 attempts. When the budget is spent the `Outcome` comes back with `ok=False`
and the failing check names, and the run continues — one unsatisfiable prompt must not
stall a batch.

**The controller composes the repair itself**, from the check names alone: *check X
failed*, appended to the prompt. No service callback. This works because a check
carries a name, so the controller can say which one failed without understanding what
it means. The give-up log and the retry message are the same fact, formatted twice.

**Belt and braces.** The prompt asks for a format; the controller verifies it. A
prompt asking nicely is not a guarantee, which is why both exist.

### Progress

After each unit, the controller accumulates `generation.usage` and `latency_ms` — facts
only the model can know — and emits `Progress(done, total, detail)`. The counting and
the estimating are the controller's, because only it knows how many units are
outstanding and whether they run sequentially or concurrently.

## Dependencies

```
service (not yet built)
   │  builds Prompts, supplies checks, consumes Outcomes
   ▼
QwenController ──────────► QwenModel ──────────► llama_cpp
   │                          │
   └──────────┬───────────────┘
              ▼
        base_model.py
   Prompt, Generation, Usage, Progress, ModelUnavailable, BaseModel
```

Edges, precisely:

- `qwen_controller` imports `Prompt`, `Generation`, `Progress`, `ModelUnavailable`
  from `base_model`, and `QwenModel` from `qwen_model`.
- `qwen_model` imports from `base_model` and lazily from `llama_cpp`. It imports
  **nothing** from `controllers`, `services` or `repositories`.
- `base_model` imports only stdlib.
- `configs/qwen.toml` is read by `qwen_model` (model, sampling) and by
  `qwen_controller` (its budget). Data, not an import.

Nothing calls up. No module here imports `engine.services` or `engine.repositories`,
so the model layer stays ignorant of purpose/risk, and the controller knows only that
checks exist and have names — never what they mean.

**One asymmetry, deliberate.** The controller imports `QwenModel` concretely rather
than `BaseModel`. That is the one-controller-per-model rule made literal: this
controller is allowed to know exactly which model it drives, and a future
`claude_controller` pairs with `claude_model`. The cost is that `QwenController`
cannot be handed a fake model for testing — see [Deferred](#deferred).

## Departures from ADR 0002

Three, to be amended in the ADR rather than left contradicted:

1. **Progress granularity** (§4, §9). The ADR says the model emits progress and the
   controller relays it. What is settled is finer: the model reports tokens and latency
   *per generation*, and the controller accumulates those into done-of-total and an
   ETA. Both layers are involved; neither duplicates the other.

2. **The model self-verifies.** The ADR gives the model no correctness check. The hello
   check puts one inside `ready()`, on the grounds that a model which can load and lie
   has not finished becoming ready.

3. **Validation with retry is controller work.** The ADR does not place it. It is the
   controller's, driven by service-supplied checks — the loop and the repair belong
   with whoever owns the loop.

## Deferred

- **The concrete `Check` shape.** It depends on what output format the prompt asks
  for, which is a prompt-engineering question not yet run. The controller only needs
  *runs against a generation* and *has a name*, so `checks` stays loosely typed until
  the format settles.
- **`fake_model.py` and `fake_controller.py`.** Stubs for testing a service that does
  not exist yet. They come back when the service does — and that is when the concrete
  `QwenModel` import above will want revisiting.

## Build order

1. `configs/qwen.toml`, and `load_defaults` in `qwen_model.py`.
2. `_check_alive` and its call from `ready()`.
3. `qwen_controller.py`: shapes, then `_run_one`, then `handle`, then `handle_async`.

Steps 1 and 2 are independently verifiable without the controller. Step 3 needs a GPU
to exercise end to end, until the fake pair exists.


## Amendment, 2026-09-23: hosting and concurrency

Built as above, then changed by measurement. `benchmarks/parallel_slots.py`
is the apparatus; the run ids below are under `data/benchmarks/`.

**`QwenModel` hosts itself on `llama-server`, not in-process `llama_cpp`.**
It starts the server in `ready()`, stops it in `close()`, and the server dies
with its Python process. One slot on the server already beat in-process
(74.6s vs 99.5s on 32 labels, run `parallel-20260923T145320Z`).

**The model runs 4 slots, and says how through two new verbs.**
`plan(prompts) -> Plan(order, too_long)` measures the set in the model's own
tokens, sets aside prompts that cannot fit a slot, and primes every slot with
the shared system turn. `submit(prompt) -> Future[Generation]` queues one
prompt; the model's pool runs at most `n_parallel` at once. The controller
submits the whole plan, handles each answer as it finishes, and re-submits a
repaired prompt at once. It never learns the slot count.

Batches with a barrier were built first and rejected: 73.5s against 73.9s one
at a time, while a continuous pool took 53.7s (`parallel-20260923T152816Z`).
This replaces `_run_one` and per-unit `Attempt` callbacks in the controller
section above; `Outcome` and the retry and repair rules are unchanged.

**Server settings were swept, not guessed** (64 labels, 4 slots, load
excluded; `parallel-20260923T154114Z`, `-155732Z`, verified end to end in the
run after them):

| setting | wall s |
|---|---|
| server defaults | 101-105 |
| MTP speculative decoding (the model's own draft head) | 77 |
| + 2,048-token ubatches + primed slots | **65.5** |
| controller → `QwenModel`, all of the above | 69.7 |

6 and 8 slots, `--kv-unified` and a 1,024 ubatch were no faster. A 2,048
ubatch without priming was slower (129s): it defeated the cache and re-read
1,460 prompt tokens a label instead of 444. Grounding held at 0.997 in every
configuration.

**`handle_async` was removed.** Parallelism is the model's decision, so an
async twin of `handle` ran the same work with no extra concurrency, and
nothing called it. An async caller runs `handle` on a worker thread:
`await asyncio.to_thread(controller.handle, prompts, checks)`.

**Departures this adds.** ADR 0001 and spec 0001 say `llama-server` is not
used; that is now superseded, on the measurement above, and those documents
still need amending. The model's interface grows from three verbs to five.

## Responsibilities, as built (2026-09-23)

What each layer does in the code today. Planned or unassigned work is not
listed here.

| # | Responsibility | Model (`QwenModel`) | Controller (`QwenController`) |
|---|---|---|---|
| 1 | Generation | Prompt in; raw text, token counts, latency and stop reason out | — |
| 2 | Prompt format | The model's chat format, thinking switched off | — |
| 3 | Constrained output | Passes a schema to the server as its grammar | — |
| 4 | Lifecycle | Starts and stops `llama-server`, loads the weights, frees the GPU | Asks the model to get ready |
| 5 | Readiness checks | Enough GPU memory, weights actually on the GPU, a coherent "Hello" | — |
| 6 | Hosting and tuning | 4 slots, speculative decoding, `-ub 2048` | — |
| 7 | Concurrency | At most 4 at once; a freed slot is refilled straight away | Submits everything, takes answers as they finish |
| 8 | Measuring the work | Counts tokens, sets aside prompts too long for a slot, gives the order, primes the slots | Asks for the plan and follows it |
| 9 | Settings | `[model]` and `[sampling]` in `configs/qwen.toml` | `[controller]`: the retry budget |
| 10 | Timeouts | 15 min for start-up, 30 min per request | — |
| 11 | Output validation | — | Runs the service's checks, by name only |
| 12 | Retries | — | Up to 3, re-submitted as soon as a check fails |
| 13 | Repair | — | Adds "previous answer failed checks X, Y" to the prompt |
| 14 | Giving up | — | `ok=False` with the reasons; the rest carries on |
| 15 | Error handling | Raises `ModelUnavailable` with the cause, naming whatever holds the GPU | Turns it into "none of the N prompts ran, because…" |
| 16 | Progress | Tokens and latency for each answer | Done/total, speed, time left, as a log line and a callback |
| 17 | Result shape | One answer per prompt | One outcome per prompt, in input order |
| 18 | Cleanup | `close()` cancels queued work; the server dies with its Python process | Cancels what it submitted if `handle` exits early |
| 19 | Logging | The server's log file | Retries, give-ups, progress |
