# 2. Engine architecture: backend layers and what each one owns

- **Status:** Proposed
- **Date:** 2026-09-17
- **Scope:** the target structure for `engine/`, designed fresh. It governs new
  work; existing code is not migrated by this document.

## Context

The pipeline built under ADR 0001 works — it extracts, validates, scores and
resumes — but it does its work **hiddenly**. Three symptoms, all the same
defect:

- **Inputs and outputs are not communicated.** `run()` takes eleven parameters
  and returns a `Path`. What is in a row, what a service hands back, what a
  driver expects — all of it is dict-shaped convention, discoverable only by
  reading the body.
- **There is no progress reporting.** A 100-label run prints lines from inside
  the loop that owns the model, the queue and the store at once. A notebook
  calling the same function gets stdout or nothing.
- **Responsibilities collect wherever the loop already is.** `run.py` is 594
  lines holding worker pool, resume, poison guard, manifest, progress and the
  per-record loop; `models/` and `controllers/` are empty directories.

The layer names below are **definitions set by the project owner** and differ
from common MVC convention deliberately. They are not re-derivable from the
names alone.

## Decision

### 1. `engine/` is the backend; frontends live outside it

`engine/` is a library. `cli.py` sits beside it, not inside it, and is one
frontend among peers — a notebook, a script or a scheduled job is the same kind
of caller with the same access.

A frontend owns argument parsing, output formatting, exit codes and display. It
owns **no workflow logic**: if a frontend has to decide something about
purpose/risk extraction, that decision is in the wrong layer.

The test of the design: *could a notebook do everything the CLI does, importing
`engine` alone?* If not, something the CLI needs is trapped inside it.

### 2. A workflow is a packaged series of services — convenience, not a wall

`engine/pipeline/` holds workflows: `start`, `resume`, `smoke`, `retry`. A
workflow is the common path through the backend, not a gate around it.

**A frontend may import and call a single service directly.** Nothing below
`pipeline/` is private by convention. A notebook that wants only scoring calls
the scoring service; it does not have to enter through a workflow to get there.

This is what decides the shape of service return values — see §6.

### 3. The layers, and what each owns

| Layer | Owns | Does not own |
|---|---|---|
| `repositories/` | the entry to get or save data | where the data physically lives — that is an implementation detail behind it |
| `models/` | an actual model (LLM, embedding, NLP): its own lifecycle **and** its hosting strategy | any knowledge of the workflow using it |
| `controllers/` | driving one model: readiness, batching, failure translation, progress relay | any data of its own |
| `services/` | the workflow's own logic — everything touching no model | how a model is hosted or executed |
| `pipeline/` | entry points that package services into a workflow | logic that belongs in a service |

**`repositories/` is the entry to get or save data.** It does not mean the data
lives there. `loader.py` is the entry to read, `store.py` the entry to write;
that today's store is JSONL run directories is behind them.

**`models/` means actual models, not data shapes.** The model loads itself,
checks its own memory, enforces one instance, closes itself. **It also owns its
hosting strategy** — in-process, `llama-server`, Ollama — and nothing above it
may branch on that. Moving `qwen_model.py` from in-process llama.cpp to
`llama-server` changes no controller and no service.

**`controllers/` are per model, not per workflow.** A controller asks its model
to be ready, translates failure into something readable, divides a work set
into batches, and catches progress from the model to write to a log. It owns
nothing. **There is no workflow-level controller** — a workflow is a workflow;
it uses services.

**Dependency direction:** `pipeline → services → controllers → models` and
`repositories`. Nothing calls up. A model never knows what workflow runs it.

### 4. Everything a layer takes and returns is explicit

This is the requirement the current code fails, so it is stated as a rule:

- **Services declare input and output as types**, not dict-shaped convention.
  A caller must be able to tell from the signature what it gets back.
- **Models emit progress.** They are the only layer that knows how far a
  generation has got.
- **Controllers relay that progress to a log.** A frontend that wants to
  display it subscribes; a frontend that does not gets a quiet library.

Progress is therefore a **callback on the model, relayed by the controller** —
not a `print` inside a loop. That is what lets one implementation serve a CLI
that overwrites a counter and a notebook that draws a bar.

### 5. Schema travels with its producer

No global `schemas/` directory. `Label` is defined in `repositories/dsld/` with
the loader that produces it. Workflow-internal shapes (`Passage`, `Claims`,
`ExtractionRow`) are defined with the service that produces them —
`passage_builder.py` produces `Passage`, `claims_parser.py` produces `Claims`.

*(Open — see §9.4: whether these are instead collected into one file per
workflow is not settled.)*

### 6. Services return values; frontends render

`scoring.py` is a service. It returns numbers.

This follows from §2: because a notebook may call the scoring service
directly, a service that returned formatted text would hand that notebook a
unified diff embedded in a string to parse its way back out of. The prompt diff
between two runs is real information, but composing it is rendering — the
service returns the two manifests and the metric delta, and `cli.py` renders
them.

### 7. No folder named for a sequence position

`preprocessing/`, `processing/`, `postprocessing/` are rejected. They say
*when*, not *what*, and stop being true the moment a workflow reorders or adds
a stage. Pre/processing remains a useful concept; it is never a folder name.

### 8. Structure

```
cli.py                          frontend: flags, formatting, exit codes
notebooks/                      frontend: peer of the CLI
engine/
├── config.py                   constants and configuration
├── repositories/
│   ├── dsld/
│   │   ├── loader.py           entry to read labels; resume lives here
│   │   ├── store.py            entry to write rows and manifests
│   │   └── schema.py           Label
│   ├── gold/
│   │   ├── loader.py           entry to read the gold set
│   │   └── schema.py           GoldLabel
│   └── prompts/
│       ├── loader.py           entry to read a prompt by id
│       ├── purpose_risk.md
│       └── purpose_risk_lines.md
├── models/
│   ├── base.py                 the model interface: ready, generate, close, progress
│   ├── schema.py               Generation, Usage, Progress
│   ├── qwen_model.py           owns llama.cpp, VRAM checks, single-instance
│   ├── claude_model.py         owns the Anthropic client
│   └── fake_model.py           no GPU: canned generations, for building above it
├── controllers/
│   ├── qwen_controller.py
│   └── claude_controller.py
├── services/
│   └── purpose_risk/
│       ├── service.py          the workflow's logic; per-record loop
│       ├── passage_builder.py  Label -> Passage
│       ├── claims_parser.py    raw text -> Claims (json and lines formats)
│       ├── boilerplate.py      claim filtering
│       ├── grounding.py        claim vs passage
│       └── scoring.py          rows + gold -> metrics; run vs run -> delta
└── pipeline/
    └── purpose_risk.py         start, resume, smoke, retry
```

### 9. Where today's `run.py` goes

| was | now | why |
|---|---|---|
| worker pool / batching | controller | only it knows what model instances exist |
| resume | `dsld/loader.py` | consults the store, returns unprocessed records only |
| progress | model emits → controller logs | the model is the only layer that knows |
| per-record loop | service | it is workflow logic |
| appending rows | `dsld/store.py` | it is the entry to save |
| poison guard | service | it is workflow policy, not model behaviour |
| manifest | `dsld/store.py` | written where rows are written |

`runner.py` as a separate module is dissolved entirely.

### 10. Concurrency is the model's decision

A service calls `controller.run(...)` or `controller.run_concurrent(...)` and
gets the same processed data back either way. The controller splits the work
into batches; **the model decides how those batches actually execute**.

Factual note: llama.cpp does support batched decode (`n_parallel`), but
`llama_cpp.Llama` — the Python wrapper used today — wraps one context with no
lock, so this repo serialises (ADR 0001 §A4). That is a fact about the current
hosting strategy, which lives inside `qwen_model.py`. Moving to `llama-server`
is a decision internal to that file and changes nothing above it.

## Consequences

**What this buys.** Every boundary states what crosses it. A model can be
swapped or rehosted without touching a service. A service can be tested against
`fake_model.py` with no GPU. Progress works the same for every frontend. The
scoring service is callable from a notebook without entering a workflow.

**What it costs.** More files than today, and more indirection between a CLI
flag and the call it causes. Two shapes — `Label` and `Passage` — that today
are the same dict. The `lines`/`json` format split, currently threaded through
`Config`, has to land somewhere explicit in `claims_parser.py`.

**What it does not do.** It does not migrate existing code. `engine/pipeline/`
as it stands today keeps working and keeps producing runs; this structure
governs what gets built next, layer by layer from the bottom.

## Build order

Bottom-up, so each layer is runnable when it is finished:

1. **`repositories/`** — loaders and stores, with schemas. No model needed.
2. **`models/`** — `base.py`, `schema.py`, then `fake_model.py`. The fake model
   is what makes everything above it buildable without a GPU.
3. **`controllers/`** — against the fake model.
4. **`services/purpose_risk/`** — the workflow logic, tested end-to-end on the
   fake model.
5. **`qwen_model.py`** — the real GPU path, dropped in behind the interface the
   fake one already proved.
6. **`pipeline/`** and **`cli.py`** — the packaging.

## Open questions

1. **Workflow-internal shapes** (§5): defined with their producer, or collected
   into one file per workflow? The "schema travels with its producer" rule
   answers this by extension, but collecting them was left deliberately open.
2. **Controller/model capacity:** does the controller ask the model how much it
   can take, or hand over the whole set and let the model chunk internally?
   Asked and not yet answered.
3. **Progress granularity:** a line per label, an overwriting counter, or a
   structured event stream written to a file? This decides whether progress is
   a callback, a log handle or an event stream — cheaper to settle before
   `models/base.py` than after.
