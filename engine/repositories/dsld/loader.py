"""The entry to read DSLD data: labels, in the two forms the engine consumes, and runs.

Being *the entry* is the whole job: callers ask this module for labels and
never open the files themselves. Where the labels physically live is an
implementation detail behind it.

**Two forms, two consumers.**

- `load_raw(dsld_id) -> dict` -- a product's full record from the
  full-database dump, every field DSLD publishes. What the cleaning service
  cleans *from*. Hand it several ids and it returns several records; name none
  and `n=` draws from the dump.
- `load_flattened(name) -> (passages_sha, {dsld_id: Label})` -- a whole
  label set the cleaning service saved: id, name and one passage each. What an
  LLM is run *over*.

They are deliberately not one function with a `kind` argument: they return
different shapes, feed different consumers, and fail for different reasons.

**Runs are read here too**, though the store writes them: `read_manifest`,
`read_ids`, `read_rows` and `done_ids`. The loader is the entry to read and
the store the entry to write, whatever the data.

**A label set is read whole, by name.** A set was chosen and written by the
cleaning service, so there is nothing to select here: which labels a run
does, and in what order, is the run's business. The set's `passages_sha`
comes back with the labels -- checked against the file, so a set edited in
place is refused -- and a run records it, so a resume can tell whether the
passages it started on are still the ones under that name.

**Ids come back sorted unless you say otherwise** on the raw side. A list of
ids is a *set*: it arrives from a CSV column, a comma-separated flag or a set
difference, where the order is incidental, so two callers asking for the same
ids should get the same answer. `order="given"` keeps caller order.

**A random draw requires a seed.** `order="random"` without one raises rather
than defaulting: an unseeded draw makes every rerun see a different subset
while looking exactly like the run before it.

**One loader instance.** A label set does not change under its name, so it is
parsed once per process and shared. `reset()` exists for tests.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path
from typing import Iterable, overload

from engine.repositories.dsld.schema import Label

REPO_ROOT = Path(__file__).resolve().parents[3]

# The full-database dump: one JSON file per product, bucketed by id.
DEFAULT_DUMP_ROOT = REPO_ROOT / "datasets" / "DSLD-full-database-JSON"

# Dump files are bucketed a thousand ids to a directory.
DUMP_BUCKET_SIZE = 1000

# Label sets, as the cleaning service saves them: labels/<name>/<file>.
LABELS_ROOT = REPO_ROOT / "data" / "dsld" / "labels"
LABEL_SET_ROWS_NAME = "passages.jsonl"
LABEL_SET_MANIFEST_NAME = "manifest.json"

# A run's files, relative to its directory.
RUN_MANIFEST = Path("manifests") / "purpose_risk.json"
RUN_IDS = Path("raw") / "ids.jsonl"
RUN_ROWS = Path("raw") / "purpose_risk.jsonl"

# How a caller may ask for records to come back.
#
# "sorted" -- ascending by id. The default when the caller names ids: a list
#             of ids is a *set*, and the order it was typed in is almost
#             always incidental -- a CSV column, a comma-separated flag, a
#             set difference. Sorting makes two callers asking for the same
#             ids get the same answer.
# "file"   -- source order, untouched. The default when reading a whole file,
#             because the pilot file is already a randomised draw written in
#             a fixed order, so a prefix of it is itself a random subset.
#             Meaningless for an id list -- there is no file to be in the
#             order of -- where "given" says the same thing honestly.
# "given"  -- caller order, for the rarer case where the list really is a
#             ranking and must survive.
# "random" -- a seeded shuffle. Never unseeded; see `_ordered`.
ORDERS = ("sorted", "file", "given", "random")


def _is_one_id(x) -> bool:
    """Is this a single id, or a collection of them?

    `str` and `bytes` are the whole reason this is a function: both are
    iterable, so a plain "is it iterable" test would read `load_raw("709")` as
    three ids -- '7', '0', '9' -- and hand back three wrong records rather
    than failing. Ids arrive as strings from CLI flags and CSV cells often
    enough that this is a certainty, not a corner case.
    """
    return isinstance(x, (int, str, bytes)) or not isinstance(x, Iterable)


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
    if order in ("file", "given"):
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


class DsldLoader:
    """The single entry to read DSLD data, in both of its forms.

    One instance per process. It holds the parsed label sets, keyed by name:
    a set does not change under its name, so a cached one is not stale. The
    dump is *not* cached: it is ~215k files and a caller reads a handful.
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
        self._flattened: dict[str, tuple[str, dict[int, Label]]] = {}

    @classmethod
    def reset(cls) -> None:
        """Drop the instance and everything it has cached.

        For tests.
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

    @overload
    def load_raw(self, dsld_id: int | str, root=..., **kw) -> dict: ...

    @overload
    def load_raw(self, dsld_id: Iterable[int | str] | None = ..., root=..., **kw) -> list[dict]: ...

    def load_raw(
        self,
        dsld_id=None,
        root: str | Path | None = None,
        n: int | None = None,
        order: str | None = None,
        seed: int | None = None,
        skip_missing: bool = False,
    ):
        """One product's full DSLD record, several, or a draw of N.

        The argument decides the shape: an id returns a `dict`, an iterable of
        ids returns a `list[dict]`. There is no separate `load_raw_many`,
        because the two would differ only in how many ids you have -- which
        the ids already say -- and a caller whose list happens to hold one
        entry should not have to switch functions to read it.

        A `str` id is deliberately *not* treated as an iterable of characters,
        which is the trap in dispatching on type: `load_raw("709")` reads
        product 709, as it always has.

        **Naming no ids draws from the dump.** `n=50` takes fifty of the
        ~215k ids the dump holds; with `order="random"` that is a sample, and
        with `order="sorted"` it is the first fifty by id. This is the entry
        for "some records" when the caller has no id list to start from --
        the population has to be listed to sample it, and only this layer
        knows how.

        **Ids come back sorted unless you say otherwise.** A list of ids is a
        *set*: it usually arrives from a CSV column, a comma-separated flag or
        a set difference, where the order is incidental. Sorting means two
        callers asking for the same ids get the same answer. `order="given"`
        keeps caller order for the rarer case where the list is a real
        ranking.

        Returned as parsed documents rather than a dataclass on purpose: the
        raw record is the *source*, and a shape defined here would quietly
        drop every field this repo does not yet consume -- which is most of
        them, and which is the reason to keep a dump at all.

        Raises `DSLDNotFound` for an id the dump does not hold, matching the
        API client, so a caller can swap transports without changing its
        error handling. In the plural form every missing id is named at once:
        a hole in a dump is usually a stale or partial download, and hearing
        about all of it beats one `DSLDNotFound` per rerun. `skip_missing`
        reads the rest instead. A draw by `n=` never reports missing ids --
        it drew them from what the dump actually holds.

        Returned as a list rather than a generator because the caller asked
        for a known set of ids, and a half-consumed generator that raises on
        the 40th is harder to reason about than one that fails up front. For a
        sweep over the whole dump, iterate `dump_ids()` and call this per id.
        """
        if dsld_id is None:
            if n is None:
                raise ValueError(
                    "load_raw needs an id, some ids, or n= to draw from the dump"
                )
            # The draw is over the ids the dump actually holds, so nothing it
            # returns can be missing -- and `n` means n records, not n
            # attempts of which some fail.
            drawn = _ordered(self.dump_ids(root), order or "sorted", seed, key=int)[:n]
            return [self._load_one_raw(i, root) for i in drawn]

        if n is not None:
            raise ValueError("n= draws from the dump; drop it, or drop the ids")

        if _is_one_id(dsld_id):
            if order is not None or seed is not None:
                raise ValueError("order= and seed= need more than one id")
            return self._load_one_raw(dsld_id, root)

        wanted = _ordered([int(i) for i in dsld_id], order or "sorted", seed, key=int)
        out, missing = [], []
        for i in wanted:
            try:
                out.append(self._load_one_raw(i, root))
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

    def _load_one_raw(self, dsld_id: int | str, root: str | Path | None) -> dict:
        path = self.dump_path(dsld_id, root)
        try:
            return json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise DSLDNotFound(
                f"{dsld_id} is not in the dump at {Path(root or DEFAULT_DUMP_ROOT)}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: {exc}") from exc

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

    def load_flattened(self, name: str) -> tuple[str, dict[int, Label]]:
        """A whole label set, and its `passages_sha`, as the cleaning service saved it.

        The labels come back keyed by id, in the set's file order. The dict is
        a copy, so a caller that edits it damages nothing; `Label` is frozen.

        The passages are hashed on read and must match the manifest's
        `passages_sha`: a set edited in place would otherwise be read as the
        set a run started on.
        """
        cached = self._flattened.get(name)
        if cached is None:
            d = Path(LABELS_ROOT) / name
            rows_p = d / LABEL_SET_ROWS_NAME
            if not rows_p.exists():
                raise FileNotFoundError(
                    f"no label set {name!r} at {d}. Label sets are saved by the "
                    f"cleaning service's save()."
                )
            manifest = json.loads((d / LABEL_SET_MANIFEST_NAME).read_text())
            passages_sha = hashlib.sha256(rows_p.read_bytes()).hexdigest()[:16]
            if passages_sha != manifest.get("passages_sha"):
                raise ValueError(
                    f"label set {name!r}: passages.jsonl hashes to {passages_sha} but its "
                    f"manifest says {manifest.get('passages_sha')} -- it was edited in "
                    f"place. Save changed passages as a new set."
                )
            labels = {}
            for record in _read_jsonl(rows_p):
                label = Label.from_dict(record)
                labels[label.dsld_id] = label
            cached = (passages_sha, labels)
            self._flattened[name] = cached
        return cached[0], dict(cached[1])


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


def load_raw(
    dsld_id=None,
    root: str | Path | None = None,
    n: int | None = None,
    order: str | None = None,
    seed: int | None = None,
    skip_missing: bool = False,
):
    """One record, several, or a draw of N. See `DsldLoader.load_raw`."""
    return loader().load_raw(
        dsld_id, root, n=n, order=order, seed=seed, skip_missing=skip_missing
    )


def dump_path(dsld_id: int | str, root: str | Path | None = None) -> Path:
    """Where `dsld_id` lives in a dump, without reading it. See `DsldLoader.dump_path`."""
    return loader().dump_path(dsld_id, root)


def dump_ids(root: str | Path | None = None) -> list[int]:
    """Every id the dump holds, ascending. See `DsldLoader.dump_ids`."""
    return loader().dump_ids(root)


def load_flattened(name: str) -> tuple[str, dict[int, Label]]:
    """A whole label set and its `passages_sha`. See `DsldLoader.load_flattened`."""
    return loader().load_flattened(name)


def reset() -> None:
    """Drop the loader and its caches. See `DsldLoader.reset`."""
    DsldLoader.reset()


# --- runs ---------------------------------------------------------------------
# Read straight from disk on every call, never cached: a run is appended to
# while it goes, so a cached read would be stale.


def read_manifest(d: str | Path) -> dict:
    """The run's manifest, or `{}` if it has none."""
    p = Path(d) / RUN_MANIFEST
    return json.loads(p.read_text()) if p.exists() else {}


def read_ids(d: str | Path) -> list[int]:
    """The run's ids, in the order the run does them."""
    return [
        int(json.loads(line)["dsld_id"])
        for line in (Path(d) / RUN_IDS).read_text().splitlines()
        if line.strip()
    ]


def read_rows(d: str | Path) -> list[dict]:
    """Read a run's rows, discarding a malformed trailing line.

    A crash mid-write leaves a partial final line. Discarding it costs the one
    record it belongs to, which resume then re-runs.
    """
    path = Path(d) / RUN_ROWS
    if not path.exists():
        return []
    rows, lines = [], path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn trailing write; the record is simply not done
            raise
    return rows


def done_ids(d: str | Path) -> set[int]:
    """IDs already present in the output -- the only progress record there is."""
    return {int(r["dsld_id"]) for r in read_rows(d) if "dsld_id" in r}
