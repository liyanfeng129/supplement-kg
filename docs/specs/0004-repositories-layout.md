# Repositories: one folder per data source, a loader to read, a store to write

- **Status:** accepted
- **Date:** 2026-10-04
- **Scope:** `engine/repositories/dsld/` (split), the dump reading in
  `engine/services/sources/dsld.py` (moved), and every caller of
  `repositories.dsld.loader` / `repositories.dsld.store`:
  `services/purpose_risk/service.py`, `services/dsld_cleaning/service.py`, the
  `test_dsld_*`, `test_store` and `test_purpose_risk_service` tests, and notebooks 04,
  06 and 08–11.

  DSLD only. PubMed (`services/sources/pubmed.py`, `data/pubmed/runs/`) is the same kind
  of misplaced repository code, but it is its own line of work: it follows in the same
  shape, as `repositories/pubmed/…`, once the DSLD folders set the pattern.

Amends ADR 0002 §8 (via ADR 0003) and spec 0003 where listed under [Departures](#departures).

## The rule

**`repositories/` has one folder per data source. In each folder, `loader.py` is the
only way to read that data and `store.py` the only way to write it.** Where the data
lives on disk is the folder's own business, written down once, in that folder.

The engine has three DSLD data sources today:

| Data source | On disk | What it is | Written by | Read by |
|---|---|---|---|---|
| raw dump | `datasets/DSLD-full-database-JSON/` | one JSON file per product, every field DSLD publishes; downloaded, never written by the engine | — | the cleaning service |
| label sets | `data/dsld/labels/<set>/` | cleaned, flattened labels ready to feed an LLM, plus annotations when a set is gold or silver | the cleaning service | the purpose/risk service |
| runs | `data/dsld/runs/<set>-<YYMMDDHHMM>-<n>/` | extraction results: one row per label | the purpose/risk service | the purpose/risk service, notebooks |

## Why

Today one `dsld/loader.py` and one `dsld/store.py` serve all three sources. Since the
loader stopped importing the store, the layout of label sets and runs (`LABELS_ROOT`,
the file names, `raw/purpose_risk.jsonl`, `manifests/purpose_risk.json`) is defined in
both modules: a rename has to be made twice, and tests patch both roots. A second,
older reader of the raw dump sits in `services/sources/dsld.py`, outside `repositories/`.

## Layout

The folders are nested by source, not flat by kind (`dsld_raw/`, `label_sets/`,
`runs/`): the names match the disk (`data/dsld/labels`, `data/dsld/runs`), "dsld" stays
the source, and PubMed later sits beside it as `repositories/pubmed/…`.

```
engine/repositories/
├── dsld/
│   ├── raw/
│   │   ├── paths.py        DUMP_ROOT, bucket size, the path of one product's file
│   │   └── loader.py       load_raw, dump_path, dump_ids          (no store: nothing writes the dump)
│   ├── labels/
│   │   ├── paths.py        LABELS_ROOT, passages.jsonl, manifest.json, annotations.jsonl
│   │   ├── schema.py       Label, GoldLabel, Claim
│   │   ├── loader.py       load_flattened
│   │   └── store.py        write_label_set
│   └── runs/
│       ├── paths.py        RUNS_ROOT, ids.jsonl, purpose_risk.jsonl, purpose_risk.json
│       ├── loader.py       read_manifest, read_ids, read_rows, todo_ids
│       └── store.py        new_run, repair, Sink
└── prompts/                unchanged
```

- `paths.py` holds locations only: constants and functions that build a `Path`. No I/O.
  It is a module of its own, not constants in the folder's `__init__.py`: explicit,
  imported by both `loader.py` and `store.py`, and the one place a test patches.
- `loader.py` and `store.py` import their folder's `paths.py`, never each other.
- No folder imports another folder's loader or store. A run refers to its label set by
  name and `passages_sha`; joining a row back to its passage is the caller's two reads.

## Responsibilities

### raw dump

- **Loader:** `load_raw` (one, many, or a seeded draw of `n`), `dump_path`, `dump_ids`.
  Unchanged from today's `dsld/loader.py`. It reads only the local dump: one adapter,
  no seam for the DSLD API.
- **No store.** The dump is downloaded, not written by the engine. The DSLD API client
  and the bulk download are download tooling, not a repository.

### label sets

- **Loader:** `load_flattened(name) -> (passages_sha, {dsld_id: Label})`. Hashes
  `passages.jsonl` on read and refuses a set whose hash no longer matches its manifest.
  Parsed once per process. Unchanged from today.
- **Store:** `write_label_set(name, rows, manifest) -> passages_sha`. A name is one set
  of passages for good: the same passages again write nothing; different passages under
  a used name are refused. Written via a temporary directory and renamed into place.
  It reads the existing manifest to compare `passages_sha`; that read serves the write
  and stays.
- **Schema:** `Label`, `GoldLabel` and `Claim` live in `labels/schema.py`. A `GoldLabel`
  is one row of a gold or silver set's `annotations.jsonl`: a label's reference claims
  (purpose and risk, each `Claim` with its evidence span) without the passage. In shape
  it is a run result; it lives with the label set because it is frozen against that
  set's passages and is what a run is evaluated against.
- **Annotations are not read here yet.** `annotations.jsonl` stays in the label set; a
  loader for it comes when scoring is built. Notebook 11 reads it directly until then.

### runs

- **Loader:** `read_manifest`, `read_ids`, `read_rows`, `todo_ids`. Read fresh on every
  call, never cached: a run is appended to while it goes. `todo_ids(run)` is the run's
  ids, in run order, without those that have a complete row: what a resume does.
- **Store:** `new_run` (ids and manifest written together, all or nothing), `repair`,
  and `Sink`.
- **No schema.** Rows stay dicts, so the store knows nothing of what produced them.

## Crashes: `repair`, the `Sink` and the loader

A crash in the middle of a flush can leave the last line of `raw/purpose_risk.jsonl`
half-written (SIGKILL, the OOM killer, power loss, a full disk; not Ctrl-C or an
exception, which close the file on a whole line). Only the last line can be torn: the
`Sink` only appends, one whole row and its newline at a time.

- **The loader tolerates exactly one thing:** a broken *last* line is skipped. So the
  label it belonged to counts as todo, and a run can be read mid-run or after a crash
  without repairing anything. A broken line anywhere else raises: it can only be
  corruption.
- **The loader never repairs.** Reading must be safe while a run is live: mid-run, the
  last line may be one the writer is still writing, and a loader that "repaired" it
  would cut a live row.
- **Repair is needed only before appending**, and appending is the store's. Appending
  after a torn line would glue the next row onto the fragment, leaving a broken line in
  the middle where the loader raises.
- **`store.repair(run)`** looks only at what follows the last newline. If it parses as
  a row (written whole, the newline lost), the newline is added; otherwise it is cut
  off. This is the loader's rule, so what the loader counted done stays done.
- **The `Sink` calls `repair` when it opens**, then buffers, checks its thresholds and
  appends (`add`, `flush`, `close`). The service never calls `repair`; a new run has
  nothing to repair.

Resume, end to end:

```
loader.read_manifest(run)            refuse if config_hash or passages_sha differ
todo = loader.todo_ids(run)          ids without a complete row (a torn last row counts as todo)
with store.Sink(run) as sink: ...    on open: repair the tail; then append fresh rows only
```

The order is free: the loader skips the torn line whether or not the `Sink` has
repaired it yet.

## `services/sources/dsld.py`: only the dump reading moves

There is no "sources" service: `services/` is workflow logic, and this folder is
repository code from before ADR 0002, never moved. This change moves the dump reading
out of it; the rest stays until its callers are retired.

| Now | Goes to |
|---|---|
| `sources/dsld.py`: `dump_path`, `load_label`, `dump_ids` | replaced by `dsld/raw/loader.py`, which does the same |
| `sources/dsld.py`: `parse_label`, its `Label` and `Ingredient` | stay, until the old pipeline and extraction code are retired |
| `sources/dsld.py`: `DSLDClient`, `fetch_products`, `save_labels`; `sources/dsld_bulk.py` | stay: download tooling, not a repository |
| `sources/pubmed.py` | stays: PubMed is out of scope |

`parse_label` and the old `Label` stay because the kept `engine/pipeline/` modules
(`render`, `config`, `validate`, `score`, `extract`) import
`services/extraction/label_purpose_risk`, which imports them; so do
`services/extraction/{gold_set,pilot_sample}.py`, two tests and three frozen
`datasets/pilot/` scripts. Retiring the old pipeline is a separate decision with its own
blast radius (the benchmark behind spec 0001's measurements, and the scripts that
rebuild the gold and silver sets).

## What moves where

A hard move: the old `repositories/dsld/loader.py`, `store.py` and `schema.py` are
deleted and every caller is updated in the same change. No re-exports are kept: the
callers are few, and re-exports would leave two import paths for one thing — the
duplication this removes.

| Today | After |
|---|---|
| `dsld/loader.py`: `load_raw`, `dump_path`, `dump_ids`, `DEFAULT_DUMP_ROOT`, `DUMP_BUCKET_SIZE` | `dsld/raw/loader.py`, `dsld/raw/paths.py` |
| `dsld/loader.py`: `load_flattened`, `LABELS_ROOT`, `LABEL_SET_*_NAME` | `dsld/labels/loader.py`, `dsld/labels/paths.py` |
| `dsld/loader.py`: `read_manifest`, `read_ids`, `read_rows`, `done_ids`, `RUN_*` | `dsld/runs/loader.py`, `dsld/runs/paths.py` |
| `dsld/store.py`: `write_label_set`, `LABELS_ROOT`, `LABEL_SET_*_NAME` | `dsld/labels/store.py`, `dsld/labels/paths.py` |
| `dsld/store.py`: `new_run`, `repair`, `Sink`, `run_id`, `rows_path`, `manifest_path`, `RUNS_ROOT` | `dsld/runs/store.py`, `dsld/runs/paths.py` |
| `dsld/schema.py`: `Label`, `GoldLabel`, `Claim` | `dsld/labels/schema.py` |

## Departures

- **ADR 0002 §8** puts runs in `dsld/store.py` and has no `repositories/runs/`. Runs
  get their own folder here, as do label sets: §8 predates label sets, and the single
  folder is what left the layout defined twice. Recorded in a new
  [ADR 0003](../adr/0003-repositories-one-folder-per-data-source/) that supersedes
  §8's repository layout; 0002 stays as the record of what was decided then.
- **Spec 0003**: the loader returns `todo_ids(run)` in place of `done_ids(run)`; the
  service no longer computes `read_ids − done_ids` itself. `store.repair` looks at the
  last line only, and keeps a whole row whose newline was lost.

## Acceptance

1. No module under `repositories/` imports another folder's loader or store; `loader.py`
   and `store.py` in the same folder import only `paths.py` (and `schema.py`).
2. Every location is defined once: a grep for `data/dsld`, `DSLD-full-database-JSON`,
   `passages.jsonl` and `purpose_risk.jsonl` under `engine/` finds only `paths.py` files.
3. Tests redirect a data source by patching its `paths.py` root alone.
4. A killed run, resumed, writes every id exactly once, and `read_rows` on the repaired
   file raises on nothing.
5. `read_rows` and `todo_ids` on an unrepaired run with a torn last line treat every
   complete row as done and the torn one as todo; with a broken line in the middle they
   raise.
6. Nothing under `repositories/` writes from a loader: `repair` is called only by the
   `Sink`.
7. Nothing imports `dump_path`, `load_label` or `dump_ids` from
   `engine.services.sources.dsld`; the dump is read only through `dsld/raw/loader.py`.
