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

- ~~**The concrete `Check` shape.**~~ Settled 2026-09-24: `Check(name, passes(prompt,
  generation), message)` in `models/base_model.py`. Checks are atomic and live with
  their prompt (`repositories/prompts/<id>/checks.py`, read by `prompts/loader.py`);
  the repair appends each failed check's `message`, and `name` is for logs only.
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

## Responsibilities, as built (2026-09-23, row 20 added the same day)

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
| 19 | Logging | The server's log file; a warning when a settings change restarts the server | Retries, give-ups, progress |
| 20 | One instance | `QwenModel()` always returns the one model; a settings change is refused mid-run and restarts the server when idle | Is handed that model |

## Amendment, 2026-09-23: one shared model, and code you read top to bottom

Settled by interview after the build above.

**`QwenModel` is a singleton.** `QwenModel(...)` is served by `__new__` and
returns the one model in the process; there is no `__init__`, because Python
would re-run it on every call and reset the shared model. A setting left as
`None` keeps its current value. A different setting is:

| model state | what happens |
|---|---|
| prompts running or queued | `RuntimeError`: settings cannot change mid-run |
| server up, idle | warning logged, server stopped, restarted with the new settings on the next `ready()` |
| server never started | the new settings are simply taken |

`close()` lets go of the model, so the next `QwenModel()` builds a fresh one
from the file. A closed model may call `ready()` again only if no other has
been built since. Two models can therefore never hold the card from one
process; another process is still caught by the free-VRAM check.

**Public surface is the five verbs.** `ready`, `generate`, `plan`, `submit`,
`close`. Everything else in `qwen_model.py` is private. Reading
`configs/qwen.toml` moved to `engine/config.py` (`read_toml`,
`QWEN_CONFIG_PATH`), shared by the model and the controller.

**Helpers only where they save a jump.** A helper stays if it is used several
times and sits next to its uses (`free_vram_mib` and `die_with_parent` are
local to `ready()`), or is used across the class (`_request`, `_render`,
`_unavailable_message`). Everything else is written where it is used:
`ready()` reads as seven numbered steps, `plan()` as three. `Usage.to_dict`,
`Generation.to_dict` and `Progress.fraction` had no callers and were removed.

**Tests call only the public verbs**, faking `nvidia-smi`, the server process
and its HTTP API underneath. The controller's scripted model is now a
`BaseModel`, since the singleton cannot be subclassed per test.

## Measurements behind the code

The numbers the code comments used to carry. Run ids are under
`data/benchmarks/`; apparatus `benchmarks/parallel_slots.py`.

| decision in the code | measured | run |
|---|---|---|
| `llama-server`, not in-process `llama_cpp` | 32 labels: 99.5s in-process vs 74.6s one server slot | `parallel-20260923T145320Z` |
| 4 slots | 32 labels: 1 slot 74.6s, 2 57.1s, **4 53.7s**, 6 55.6s, 8 56.2s -- prompt reading already runs at full width, so more slots only queue | `parallel-20260923T145320Z` |
| continuous pool, no batch barrier | 73.5s batched vs 73.9s one at a time vs 53.7s pool | `parallel-20260923T152816Z` |
| MTP speculative decoding, 3 drafted tokens | 64 labels: 101-105s defaults -> 77s; 5 drafted tokens 2.5% faster (noise) for 1.2 GB more VRAM | sweep `parallel-20260923T154114Z`, `-155732Z` |
| `-ub 2048` + primed slots | 77s -> 65.5s; prompt tokens read per label 1,460 -> 444. Without priming, 2,048 was slower (129s) | sweep `parallel-20260923T154114Z`, `-155732Z` |
| `--no-kv-unified` | `--kv-unified` changed nothing; per-slot context makes `plan()`'s fit check exact | sweep `parallel-20260923T154114Z`, `-155732Z` |
| input order, not longest-first | longest-first read 32,371 prompt tokens vs 25,317: its first requests missed the cached system prompt | -- |
| `min_vram_mib = 21500` | peak ~20.9 GB at 4 slots x 5,120 tokens with the settings above | -- |
| JSON schema costs ~4x decode | grammar sampler vs free sampling | -- |
