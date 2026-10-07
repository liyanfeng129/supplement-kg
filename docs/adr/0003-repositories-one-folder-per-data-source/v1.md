# 3. Repositories: one folder per data source

- **Status:** Accepted
- **Date:** 2026-10-05
- **Supersedes:** ADR 0002 §8 (the `repositories/` part of the structure) and the
  *resume* row of §9 where it has the loader consult the store. The rest of 0002 stands.
- **Spec:** [0004 — repositories layout](../specs/0004-repositories-layout.md)

## Context

ADR 0002 §8 gave DSLD one repository folder, `repositories/dsld/`, with one `loader.py`
and one `store.py`, plus a separate `repositories/gold/` for the gold set. Two things
came after it:

- **Label sets.** `data/dsld/labels/<set>/` holds cleaned passages, written by the
  cleaning service and read by the purpose/risk service. Gold and silver sets carry
  their reference claims in the same folder, as `annotations.jsonl`.
- **Runs keyed to label sets.** A run refers to its set by name and `passages_sha`.

So one `dsld/loader.py` and one `dsld/store.py` now serve three data sources: the raw
dump, label sets and runs. Once the loader stopped importing the store, each module
defined the layout of label sets and runs for itself (roots, file names). A rename had to
be made twice, and tests patched both roots. A `repositories/gold/` folder would add a
second repository reading the label-set folders.

## Decision

**`repositories/` has one folder per data source. In each folder, `loader.py` is the
only way to read that data and `store.py` the only way to write it. Where the data
lives is written once, in the folder's `paths.py`.**

```
engine/repositories/
├── dsld/
│   ├── raw/       paths.py, loader.py                         (no store: the dump is downloaded)
│   ├── labels/    paths.py, schema.py, loader.py, store.py    Label, GoldLabel, Claim
│   └── runs/      paths.py, loader.py, store.py               rows stay dicts
└── prompts/
```

1. **`paths.py` holds locations only:** constants and functions that build a `Path`, no
   I/O. `loader.py` and `store.py` import it and never each other. Tests redirect a data
   source by patching its `paths.py` alone.
2. **No folder imports another folder's loader or store.** Joining a run row to its
   passage is two reads by the caller.
3. **There is no `repositories/gold/`.** Gold and silver annotations are part of a label
   set, so `GoldLabel` and `Claim` live in `dsld/labels/schema.py`. A `GoldLabel` is a
   label's reference claims without the passage: a run result in shape, frozen against
   that set's passages, and what runs are evaluated against.
4. **The loader says what is left; the store prepares to append.** 0002 §9 had the
   loader "consult the store and return unprocessed records". The loader still returns
   them, as `todo_ids(run)`, but from the run's own files: it never imports the store.
   A torn last line counts as todo. Repairing that line is a write, needed only before
   appending, so the `Sink` does it when it opens; a loader that repaired would cut the
   row a live run is still writing.
5. **A store may read to serve its write.** `store.repair` reads the last line it may
   cut, and `write_label_set` reads the manifest it may refuse to overwrite. Reading
   data *for a caller* stays the loader's job alone.

## Consequences

- More files than 0002 drew, but every location is defined once and each folder can be
  tested and changed alone.
- Callers of `repositories.dsld.loader` / `store` move to the new paths in one hard move;
  no re-exports are kept.
- PubMed follows the same shape later (`repositories/pubmed/…`); it is not part of this
  decision.
- 0002 is left as written, with a note pointing here, as the record of what was decided
  then.
