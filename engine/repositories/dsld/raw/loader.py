"""The entry to read the full-database dump: one product's full DSLD record, or many.

- `load_raw(dsld_id) -> dict` -- a product's full record, every field DSLD
  publishes. What the cleaning service cleans *from*. Hand it several ids and
  it returns several records; name none and `n=` draws from the dump.
- `dump_path(dsld_id)` -- where a record lives, without reading it.
- `dump_ids()` -- every id the dump holds.

Only the local dump is read: the DSLD API is download tooling, not a second
source behind this loader. There is no store beside it, because the engine
never writes the dump. Where the dump lives is `paths.py`'s business.

**Ids come back sorted unless you say otherwise.** A list of ids is a *set*:
it arrives from a CSV column, a comma-separated flag or a set difference,
where the order is incidental, so two callers asking for the same ids should
get the same answer. `order="given"` keeps caller order.

**A random draw requires a seed.** `order="random"` without one raises rather
than defaulting: an unseeded draw makes every rerun see a different subset
while looking exactly like the run before it.

Nothing is cached: the dump is ~215k files and a caller reads a handful.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, overload

from engine.repositories.dsld.raw import paths

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


@overload
def load_raw(dsld_id: int | str, root=..., **kw) -> dict: ...


@overload
def load_raw(dsld_id: Iterable[int | str] | None = ..., root=..., **kw) -> list[dict]: ...


def load_raw(
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
        drawn = _ordered(dump_ids(root), order or "sorted", seed, key=int)[:n]
        return [_load_one_raw(i, root) for i in drawn]

    if n is not None:
        raise ValueError("n= draws from the dump; drop it, or drop the ids")

    if _is_one_id(dsld_id):
        if order is not None or seed is not None:
            raise ValueError("order= and seed= need more than one id")
        return _load_one_raw(dsld_id, root)

    wanted = _ordered([int(i) for i in dsld_id], order or "sorted", seed, key=int)
    out, missing = [], []
    for i in wanted:
        try:
            out.append(_load_one_raw(i, root))
        except DSLDNotFound:
            missing.append(i)
    if missing and not skip_missing:
        shown = ", ".join(str(i) for i in missing[:5])
        more = f" (+{len(missing) - 5})" if len(missing) > 5 else ""
        raise DSLDNotFound(
            f"{len(missing)} of {len(wanted)} id(s) are not in the dump at "
            f"{Path(root or paths.DUMP_ROOT)}: {shown}{more}. "
            f"Pass skip_missing=True to read the rest."
        )
    return out


def _load_one_raw(dsld_id: int | str, root: str | Path | None) -> dict:
    path = paths.dump_path(dsld_id, root)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise DSLDNotFound(
            f"{dsld_id} is not in the dump at {Path(root or paths.DUMP_ROOT)}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def dump_path(dsld_id: int | str, root: str | Path | None = None) -> Path:
    """Where `dsld_id` lives in a full-database dump. No existence check."""
    return paths.dump_path(dsld_id, root)


def dump_ids(root: str | Path | None = None) -> list[int]:
    """Every id the dump holds, ascending.

    The entry for "many, in some order, randomly" over the *raw* side:
    there is no file listing the dump's contents, so without this a
    caller cannot ask for a sample of it at all.
    """
    out: list[int] = []
    for bucket in Path(root or paths.DUMP_ROOT).iterdir():
        if not bucket.is_dir():
            continue
        for f in bucket.iterdir():
            if f.suffix == ".json" and f.stem.isdigit():
                out.append(int(f.stem))
    return sorted(out)
