"""The entry to read DSLD labels.

Being *the entry* is the whole job: callers ask this module for labels and
never open the file themselves. Where the labels physically live -- a frozen
pilot JSONL today, a database or the DSLD API later -- is an implementation
detail behind it, and moving them changes this file and nothing above it.

**Three sources, three named entries.** A DSLD label exists here in three
forms, and a caller that has to remember which file holds which is not being
given an entry at all:

- `load_raw(dsld_id) -> dict` -- **one product**, its full record from the
  full-database dump, every field DSLD publishes. What a passage is built
  *from*.
- `load_flattened() -> list[Label]` -- **the whole corpus** of frozen pilot
  passages: id, name, one rendered string each. What an extractor is run
  *over*.
- `load_gold() -> list[GoldLabel]` -- **the whole reference set**, the
  purpose/risk reading of 100 of those labels. What a run is scored *against*.

Note the first is singular and the other two are plural, which is a fact about
the sources rather than an inconsistency: the dump is ~215k separate JSON files
and a caller reads a handful, so there is no whole to load. The pilot and the
gold set are one file each, of 1000 and 100 lines -- small enough to read once
and hold. That is also why only the latter two are cached.

They are deliberately not one function with a `kind` argument: they return
different shapes (`dict`, `Label`, `GoldLabel`), take different arguments, and
fail for different reasons.

**One, many, in order, or at random.** Every entry answers all four, and the
raw side has `load_raw_many()` and `dump_ids()` so that "a random sample of
the dump" is expressible at all -- there is no file listing the dump, so
without the latter a caller cannot even name the population to sample from.

    load_raw(709)                                   # one
    load_raw_many([889, 709])                       # many, caller order
    load_flattened(order="sorted")                  # all, by id
    load_flattened(order="random", seed=42)         # all, shuffled
    load(limit=50, order="random", seed=42)         # a random 50

**A random draw requires a seed.** `order="random"` without one raises rather
than defaulting, because the failure is otherwise silent: an unseeded draw
makes every rerun score a different subset of labels while looking exactly
like the run before it. This follows the rule `pilot_sample.sample` already
set. `ids=` is itself an order, so combining it with `order=` also raises
instead of quietly discarding one of the two.

**One loader instance.** The three sources are read repeatedly -- a scoring
pass wants the gold set, then the passages that produced it, then a run's rows
-- and re-reading a 1000-line JSONL per call is both slow and a source of
disagreement if the file changes underneath a session. `DsldLoader` enforces a
single instance so each parsed file is read once and shared; `reset()` exists
for tests and for a caller that has genuinely rewritten a source.

**Resume lives here**, because resume is a question about what is left to read:
the store knows which ids are done, so the loader consults it and hands back
only the rest. A caller that wants everything simply does not pass a run.

**A selection is returned, not a bare list.** The old loader returned records
and printed the counts it had dropped, which meant a notebook calling it either
scraped stdout or guessed. `Selection` carries the numbers that were previously
only printed -- how many matched, how many were already done, which requested
ids do not exist -- so the caller can report them however it likes. That is the
"say what is in and out" rule from ADR 0002 §4 applied to the one layer that
knows.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path

from engine.repositories.dsld import store
from engine.repositories.dsld.schema import GoldLabel, Label

REPO_ROOT = Path(__file__).resolve().parents[3]

# The frozen pilot passages. The gold set was labelled against these exact
# strings, so a run that re-renders instead of reading them is not comparable
# to the gold set -- see `schema.Label`.
DEFAULT_RECORDS = (
    REPO_ROOT / "data" / "dsld" / "runs" / "20260908T125504Z" / "raw" / "pilot-1000.jsonl"
)

# The full-database dump: one JSON file per product, bucketed by id.
DEFAULT_DUMP_ROOT = REPO_ROOT / "datasets" / "DSLD-full-database-JSON"

# The gold set. Committed outside the gitignored `data/` tree on purpose: it
# cost money to produce and every later comparison depends on it, so clearing
# a cache must not destroy it.
DEFAULT_GOLD = REPO_ROOT / "datasets" / "pilot" / "gold-100.jsonl"

# Dump files are bucketed a thousand ids to a directory.
DUMP_BUCKET_SIZE = 1000

# How a caller may ask for records to come back.
#
# "file"   -- source order, untouched. The default everywhere, because the
#             pilot file is already a randomised draw written in a fixed
#             order, so a prefix of it is itself a random subset.
# "sorted" -- ascending by id. For a stable order across sources that were
#             written in different orders.
# "random" -- a seeded shuffle. Never unseeded; see `_ordered`.
ORDERS = ("file", "sorted", "random")


def _ordered(items: list, order: str, seed: int | None, key) -> list:
    """Apply an ordering, refusing a random draw that could not be redrawn.

    The seed is required rather than defaulted. `pilot_sample.sample` sets the
    rule this follows -- sorted input plus an explicit seed, so the same
    (source, seed, n) redraws the same set -- and the reason is that the
    alternative is silent: an unseeded draw makes every rerun score a
    different subset of labels while looking exactly like the run before it.
    Defaulting the seed would hide *which* seed produced a result, which is
    the same problem one step later, when a manifest has to record it.

    The shuffle is applied to a sorted copy, not to file order, so the draw
    depends only on the seed and the set of ids -- not on how the file
    happened to be written.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, not {order!r}")
    if order == "file":
        if seed is not None:
            raise ValueError("seed only means something with order='random'")
        return items
    if order == "sorted":
        if seed is not None:
            raise ValueError("seed only means something with order='random'")
        return sorted(items, key=key)
    if seed is None:
        raise ValueError(
            "a random draw needs an explicit seed, or it cannot be redrawn: "
            "pass seed=... (any int). An unseeded draw would make every rerun "
            "score a different subset while looking identical."
        )
    out = sorted(items, key=key)
    random.Random(seed).shuffle(out)
    return out


class DSLDNotFound(LookupError):
    """No record exists for the requested id.

    Named to match what the DSLD API client raises for an id the API does not
    know, so a caller can swap a local dump for the API without changing its
    error handling.
    """


@dataclass(frozen=True)
class Selection:
    """Labels to process, and an account of everything left out.

    The account is the point. `labels` alone cannot distinguish "you asked for
    500 and the file holds 200" from "you asked for 500 and 300 were already
    done", and those call for opposite responses from the caller.
    """

    labels: list[Label]
    n_available: int = 0
    n_already_done: int = 0
    missing_ids: list[int] = field(default_factory=list)
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    @property
    def ids(self) -> list[int]:
        return [lab.dsld_id for lab in self.labels]

    def summary(self) -> str:
        """One line a frontend can print as-is, or ignore in favour of the fields."""
        parts = [f"{len(self.labels)} to do of {self.n_available} available"]
        if self.n_already_done:
            parts.append(f"{self.n_already_done} already done")
        if self.missing_ids:
            shown = ", ".join(str(i) for i in self.missing_ids[:5])
            more = f" (+{len(self.missing_ids) - 5})" if len(self.missing_ids) > 5 else ""
            parts.append(f"{len(self.missing_ids)} requested id(s) not in source: {shown}{more}")
        return "; ".join(parts)


class DsldLoader:
    """The single entry to read DSLD data, in any of its three forms.

    One instance per process. The instance holds the parsed corpora so a
    session that reads the passages, then the gold set, then the passages
    again pays for each file once and cannot see two different versions of
    one file within a run.

    Caching is keyed by resolved path, so passing an explicit path -- a test
    fixture, a second pilot draw -- is a normal call and not a cache bypass.
    The dump is *not* cached: it is ~92k files and a caller reads a handful.
    """

    _instance: "DsldLoader | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "DsldLoader":
        # Double-checked under a lock: the controllers layer may ask for the
        # loader from more than one worker thread, and two instances would
        # mean two caches and two answers to the same question.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._flattened: dict[Path, list[Label]] = {}
        self._gold: dict[Path, list[GoldLabel]] = {}

    @classmethod
    def reset(cls) -> None:
        """Drop the instance and everything it has cached.

        For tests, and for a caller that has just rewritten a source file and
        means the next read to see it.
        """
        with cls._lock:
            cls._instance = None

    # --- raw ----------------------------------------------------------------

    def dump_path(self, dsld_id: int | str, root: str | Path | None = None) -> Path:
        """Where `dsld_id` lives in a full-database dump. No existence check."""
        i = int(dsld_id)
        lo = (i // DUMP_BUCKET_SIZE) * DUMP_BUCKET_SIZE
        bucket = f"{lo:06d}-{lo + DUMP_BUCKET_SIZE - 1:06d}"
        return Path(root or DEFAULT_DUMP_ROOT) / bucket / f"{i}.json"

    def load_raw(self, dsld_id: int | str, root: str | Path | None = None) -> dict:
        """One product's full DSLD record, exactly as published.

        Returned as the parsed document rather than a dataclass on purpose:
        the raw record is the *source*, and a shape defined here would quietly
        drop every field this repo does not yet consume -- which is most of
        them, and which is the reason to keep a dump at all.

        Raises `DSLDNotFound` for an id the dump does not hold, matching the
        API client, so a caller can swap transports without changing its
        error handling.
        """
        path = self.dump_path(dsld_id, root)
        try:
            return json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise DSLDNotFound(
                f"{dsld_id} is not in the dump at {Path(root or DEFAULT_DUMP_ROOT)}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: {exc}") from exc

    def load_raw_many(
        self,
        ids,
        root: str | Path | None = None,
        order: str = "file",
        seed: int | None = None,
        skip_missing: bool = False,
    ) -> list[dict]:
        """Several products' full records.

        `order="file"` means caller order here -- there is no file to be in
        the order of, and an explicit id list is usually a priority order.

        Returned as a list rather than a generator because the caller asked
        for a known set of ids and a half-consumed generator that raises on
        the 40th id is harder to reason about than one that fails up front.
        For a sweep over the whole dump, iterate `dump_ids()` and call
        `load_raw` per id instead -- that is the case a list would blow up on.

        `skip_missing` is off by default: a dump with a hole in it is usually
        a stale or partial download, and discovering that at the end of a
        batch beats discovering it in the output.
        """
        wanted = _ordered([int(i) for i in ids], order, seed, key=int)
        out, missing = [], []
        for i in wanted:
            try:
                out.append(self.load_raw(i, root))
            except DSLDNotFound:
                missing.append(i)
        if missing and not skip_missing:
            shown = ", ".join(str(i) for i in missing[:5])
            more = f" (+{len(missing) - 5})" if len(missing) > 5 else ""
            raise DSLDNotFound(
                f"{len(missing)} of {len(wanted)} id(s) are not in the dump at "
                f"{Path(root or DEFAULT_DUMP_ROOT)}: {shown}{more}. "
                f"Pass skip_missing=True to read the rest."
            )
        return out

    def dump_ids(self, root: str | Path | None = None) -> list[int]:
        """Every id the dump holds, ascending.

        The entry for "many, in some order, randomly" over the *raw* side:
        there is no file listing the dump's contents, so without this a
        caller cannot ask for a sample of it at all.
        """
        out: list[int] = []
        for bucket in Path(root or DEFAULT_DUMP_ROOT).iterdir():
            if not bucket.is_dir():
                continue
            for f in bucket.iterdir():
                if f.suffix == ".json" and f.stem.isdigit():
                    out.append(int(f.stem))
        return sorted(out)

    # --- flattened ----------------------------------------------------------

    def load_flattened(
        self,
        path: str | Path | None = None,
        order: str = "file",
        seed: int | None = None,
    ) -> list[Label]:
        """Every frozen pilot passage, in file order unless asked otherwise.

        These are read as data and never re-rendered: the gold set was
        labelled against these exact strings, so a run that rebuilds a passage
        from the dump is not comparable to it.

        Separate from `load()` so that counting or inspecting the corpus does
        not require constructing a selection, and so `load()` has one obvious
        place to get its universe from.
        """
        src = Path(path or DEFAULT_RECORDS).resolve()
        cached = self._flattened.get(src)
        if cached is None:
            if not src.exists():
                raise FileNotFoundError(
                    f"no label records at {src}. The pilot passages are gitignored "
                    f"(raw/ is not committed); re-run the pilot or point --records elsewhere."
                )
            cached = [Label.from_dict(r) for r in _read_jsonl(src)]
            self._flattened[src] = cached
        # Ordering is applied to a copy on the way out, never to the cache:
        # the cache stays in canonical file order, so one caller's random
        # draw cannot change what the next caller sees. The copy also means a
        # caller that sorts in place damages nothing -- `Label` is frozen, so
        # the elements themselves need no copying.
        return _ordered(list(cached), order, seed, key=lambda lab: lab.dsld_id)

    # --- gold ---------------------------------------------------------------

    def load_gold(
        self,
        path: str | Path | None = None,
        order: str = "file",
        seed: int | None = None,
    ) -> list[GoldLabel]:
        """The reference purpose/risk reading, in file order unless asked otherwise.

        A reference, not ground truth: one model's reading of each passage,
        auditable because every claim keeps the evidence span it came from.
        """
        src = Path(path or DEFAULT_GOLD).resolve()
        cached = self._gold.get(src)
        if cached is None:
            if not src.exists():
                raise FileNotFoundError(
                    f"no gold set at {src}. It is committed under datasets/ rather "
                    f"than data/; if it is absent, rebuild it with "
                    f"services/extraction/gold_set.py -- it calls the frontier model."
                )
            cached = [GoldLabel.from_dict(r) for r in _read_jsonl(src)]
            self._gold[src] = cached
        return _ordered(list(cached), order, seed, key=lambda g: g.dsld_id)

    def gold_by_id(self, path: str | Path | None = None) -> dict[int, GoldLabel]:
        """The gold set keyed by id -- what scoring joins a run's rows against."""
        return {g.dsld_id: g for g in self.load_gold(path)}

    # --- selection ----------------------------------------------------------

    def load(
        self,
        path: str | Path | None = None,
        ids: list[int] | None = None,
        limit: int | None = None,
        resume_from: str | Path | None = None,
        order: str = "file",
        seed: int | None = None,
    ) -> Selection:
        """Flattened labels to process, with an account of what was left out.

        `ids` selects specific labels; ids absent from the source are reported
        in `Selection.missing_ids` rather than silently yielding a shorter
        list, since a typo'd id and a genuinely absent one look identical in a
        bare result.

        `limit` applies after id selection and after the resume filter, so
        `limit=10` means ten labels that will actually be processed -- not ten
        candidates of which some are already done. The old loader applied it
        before the resume filter, where `--limit 10` on a resumed run could
        hand back nothing at all and look like completion.

        `resume_from` is a run directory. Labels already present in its rows
        are dropped and counted. It is explicit and never inferred: appending
        to an existing run is a deliberate act.
        """
        src = Path(path or DEFAULT_RECORDS)
        labels = self.load_flattened(src)
        n_available = len(labels)

        # Ordering happens before id selection so that an explicit id list
        # still wins: `ids` is a priority order, and shuffling after it would
        # throw that away. With `ids`, this orders nothing the caller sees.
        if ids is None:
            labels = _ordered(labels, order, seed, key=lambda lab: lab.dsld_id)
        elif order != "file" or seed is not None:
            raise ValueError(
                "ids= already states an order; drop order=/seed= or drop ids="
            )

        missing: list[int] = []
        if ids is not None:
            wanted = [int(i) for i in ids]
            by_id = {lab.dsld_id: lab for lab in labels}
            missing = [i for i in wanted if i not in by_id]
            # Caller order, not file order: an explicit id list is usually a
            # priority order, and silently re-sorting it hides that it was ignored.
            labels = [by_id[i] for i in wanted if i in by_id]

        n_done = 0
        if resume_from is not None:
            done = store.done_ids(resume_from)
            if done:
                before = len(labels)
                labels = [lab for lab in labels if lab.dsld_id not in done]
                n_done = before - len(labels)

        if limit is not None:
            labels = labels[:limit]

        return Selection(
            labels=labels,
            n_available=n_available,
            n_already_done=n_done,
            missing_ids=missing,
            source=src,
        )


def _read_jsonl(src: Path) -> list[dict]:
    """Parse a JSONL file, naming the line number of a bad record.

    A corpus file is edited by hand often enough that "line 412" is the
    difference between a one-minute fix and a bisect.
    """
    out = []
    with src.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{src}:{n}: {exc}") from exc
    return out


def loader() -> DsldLoader:
    """The one loader. Cheap to call; every call returns the same instance."""
    return DsldLoader()


# Module-level entries. The class is the thing, but a caller reading labels
# should not have to know it exists -- `loader.load_flattened()` is the entry,
# and that it is backed by a shared instance is this file's business.


def load_raw(dsld_id: int | str, root: str | Path | None = None) -> dict:
    """One product's full DSLD record. See `DsldLoader.load_raw`."""
    return loader().load_raw(dsld_id, root)


def dump_path(dsld_id: int | str, root: str | Path | None = None) -> Path:
    """Where `dsld_id` lives in a dump, without reading it. See `DsldLoader.dump_path`."""
    return loader().dump_path(dsld_id, root)


def load_raw_many(
    ids,
    root: str | Path | None = None,
    order: str = "file",
    seed: int | None = None,
    skip_missing: bool = False,
) -> list[dict]:
    """Several products' full records. See `DsldLoader.load_raw_many`."""
    return loader().load_raw_many(
        ids, root, order=order, seed=seed, skip_missing=skip_missing
    )


def dump_ids(root: str | Path | None = None) -> list[int]:
    """Every id the dump holds, ascending. See `DsldLoader.dump_ids`."""
    return loader().dump_ids(root)


def load_flattened(
    path: str | Path | None = None, order: str = "file", seed: int | None = None
) -> list[Label]:
    """The frozen pilot passages. See `DsldLoader.load_flattened`."""
    return loader().load_flattened(path, order=order, seed=seed)


def load_gold(
    path: str | Path | None = None, order: str = "file", seed: int | None = None
) -> list[GoldLabel]:
    """The reference purpose/risk reading. See `DsldLoader.load_gold`."""
    return loader().load_gold(path, order=order, seed=seed)


def gold_by_id(path: str | Path | None = None) -> dict[int, GoldLabel]:
    """The gold set keyed by id. See `DsldLoader.gold_by_id`."""
    return loader().gold_by_id(path)


def load(
    path: str | Path | None = None,
    ids: list[int] | None = None,
    limit: int | None = None,
    resume_from: str | Path | None = None,
    order: str = "file",
    seed: int | None = None,
) -> Selection:
    """Labels to process, with an account of what was left out. See `DsldLoader.load`."""
    return loader().load(
        path, ids=ids, limit=limit, resume_from=resume_from, order=order, seed=seed
    )


def read_all(path: str | Path | None = None) -> list[Label]:
    """Deprecated alias for `load_flattened`.

    Kept because the pilot notebook and CLI call it; the name said nothing
    about *which* of the three forms it read, which is the ambiguity the three
    named entries exist to remove.
    """
    return load_flattened(path)


def reset() -> None:
    """Drop the loader and its caches. See `DsldLoader.reset`."""
    DsldLoader.reset()
