"""The entry to save DSLD data: label sets and runs, as JSONL under `data/dsld/`.

**A label set is written once under a name that says what it is.** The
cleaning service saves its flattened labels with `write_label_set`:
`labels/<name>/passages.jsonl` plus a `manifest.json` that records, among
what the set is, `passages_sha` -- a hash of the passages. A name means one
set of passages for good: saving the same passages again writes nothing, and
saving different ones under a used name is refused. A set may also hold the
annotations made on it (`annotations.jsonl`), which is what makes it a gold or
silver set. A run refers to its labels by name and `passages_sha` and never
copies them, so a result can always be joined back to the passage it was read
from.

**A run is created whole, then only appended to.** `new_run` writes
`runs/<set name>-<YYMMDDHHMM>-<n>/raw/ids.jsonl` -- the run's ids, in run
order -- and its manifest together, so a crash leaves the whole run or none of
it. Results are appended to `raw/purpose_risk.jsonl` by a `Sink`. Every row
carries `config_hash` inline, so a row is interpretable without joining it
back to a manifest.

**Done is read from the results, never stored.** A label is done when its id
is in the results file; `loader.done_ids` reads it, as this module only
writes. There is no status field to update: JSONL cannot change a line in
place, and a status beside a result is two facts that a crash can leave
disagreeing. A malformed trailing line -- a crash mid-write -- is discarded
and that label re-run, costing one label rather than corrupting a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from engine import config
from engine.repositories.dsld.schema import Label

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO_ROOT / "data" / "dsld" / "runs"
LABELS_ROOT = REPO_ROOT / "data" / "dsld" / "labels"

ROWS_NAME = "purpose_risk.jsonl"
MANIFEST_NAME = "purpose_risk.json"
IDS_NAME = "ids.jsonl"

LABEL_SET_ROWS_NAME = "passages.jsonl"
LABEL_SET_MANIFEST_NAME = "manifest.json"
LABEL_SET_ANNOTATIONS_NAME = "annotations.jsonl"

# Used when `configs/engine.toml` has no `[store]` value.
_FALLBACK_FLUSH_EVERY = 64
_FALLBACK_FLUSH_SECONDS = 30.0


def run_id() -> str:
    """The current UTC minute, `YYMMDDHHMM`: the time part of a run's name."""
    return datetime.now(timezone.utc).strftime("%y%m%d%H%M")


def rows_path(d: str | Path, name: str = ROWS_NAME) -> Path:
    """Where a run's rows live. `name` selects which artifact in the run."""
    return Path(d) / "raw" / name


def manifest_path(d: str | Path, name: str = MANIFEST_NAME) -> Path:
    """Where a run's manifest lives. `name` selects which artifact in the run."""
    return Path(d) / "manifests" / name


def repair(d: str | Path) -> int:
    """Rewrite the rows file without its torn trailing line, if any.

    Returns the number of rows kept. Appending after a torn line would leave
    one unparseable line in the middle of the file, where `loader.read_rows`
    cannot safely skip it.
    """
    path = rows_path(d)
    if not path.exists():
        return 0
    # The rule `loader.read_rows` reads by, applied here to rewrite: a bad
    # last line is a torn write and dropped; a bad line anywhere else raises.
    rows, lines = [], path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


class Sink:
    """Where a streaming run's rows go: buffered, then appended in batches.

    `add` costs a list append and two comparisons, so it is safe on the loop
    that feeds the model. A row reaches disk when `flush_every` rows have
    queued up or `flush_seconds` have passed since the last flush -- whichever
    comes first -- and a crash loses only what is still in the buffer, which
    resume re-runs.

    `flush` appends and fsyncs; it never rewrites, so the torn-trailing-line
    rule in `loader.read_rows` still covers a crash mid-flush. Opening a sink
    on an existing run repairs it once first, for the reason `repair` exists.

    A context manager, so `close()` runs on a normal exit, on an exception and
    on Ctrl-C. Flushes run on the caller's thread: a local append of 64 rows
    takes milliseconds against seconds per generation.

    Both thresholds come from `configs/engine.toml` `[store]` unless passed; a
    missing or non-numeric value falls back to 64 rows and 30 seconds. Rows
    are plain dicts: the sink knows nothing of what produced them.
    """

    def __init__(
        self,
        run: str | Path,
        flush_every: int | None = None,
        flush_seconds: float | None = None,
    ):
        table = config.read_toml(config.ENGINE_CONFIG_PATH).get("store")
        table = table if isinstance(table, dict) else {}
        every, seconds = table.get("flush_every"), table.get("flush_seconds")
        if not isinstance(every, int) or isinstance(every, bool):
            every = _FALLBACK_FLUSH_EVERY
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            seconds = _FALLBACK_FLUSH_SECONDS
        self.path = rows_path(run)
        self.flush_every = every if flush_every is None else flush_every
        self.flush_seconds = float(seconds) if flush_seconds is None else flush_seconds
        self._buffer: list[dict] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair(run)
        self._last_flush = time.monotonic()

    def add(self, row: dict) -> None:
        """Buffer one row; flush if either threshold is reached."""
        self._buffer.append(row)
        if (
            len(self._buffer) >= self.flush_every
            or time.monotonic() - self._last_flush >= self.flush_seconds
        ):
            self.flush()

    def flush(self) -> None:
        """Append every buffered row to the run's rows file and fsync it."""
        if self._buffer:
            with self.path.open("a") as fh:
                for r in self._buffer:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._buffer = []
        self._last_flush = time.monotonic()

    def close(self) -> None:
        """Flush what is left. Idempotent."""
        self.flush()

    def __enter__(self) -> "Sink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def new_run(name: str, ids: list[int], manifest: dict, root: str | Path | None = None) -> Path:
    """Create a run over label set `name`: its directory, ids and manifest, all or nothing.

    The directory is `<name>-<YYMMDDHHMM>-<n>`, the time in UTC and `n`
    counting from 0 for runs started in the same minute. Everything is written
    into a hidden temporary directory beside the runs and renamed into place
    last, so a crash leaves either a complete run or no run -- never a
    manifest without ids, which would look like a run with nothing to do, nor
    ids without a manifest.
    """
    root = Path(root or RUNS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".new-run-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        (tmp / "raw").mkdir()
        (tmp / "manifests").mkdir()
        with rows_path(tmp, IDS_NAME).open("w") as fh:
            for i in ids:
                fh.write(json.dumps({"dsld_id": int(i)}) + "\n")
        manifest_path(tmp).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        stamp = run_id()
        n = 0
        while True:
            d = root / f"{name}-{stamp}-{n}"
            if not d.exists():
                try:
                    os.rename(tmp, d)
                    break
                except OSError:
                    if not d.exists():
                        raise
            n += 1  # taken, perhaps by a run started a moment ago: the next number
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return d


def write_label_set(
    name: str, rows: list[dict], manifest: dict, root: str | Path | None = None
) -> str:
    """Save flattened labels as label set `name`; return its `passages_sha`.

    `passages_sha` is the first 16 hex digits of the sha256 of
    `passages.jsonl` as written. It goes into the manifest beside `name`, so
    the manifest passed in describes the set and need not carry either.

    A name is one set of passages for good. If `name` exists with the same
    `passages_sha`, nothing is written and its manifest is kept; if it exists
    with a different one, this refuses -- changed passages need a new name, or
    every run that read the old ones would silently point at the new.

    Every row must read as a `Label` and ids must be unique: a set is looked up
    by id, and a duplicate would silently shadow the other row. A new set is
    written into a hidden temporary directory and renamed into place, so a
    half-written set never exists under a name.
    """
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"{name!r} cannot name a label set directory")
    seen: set[int] = set()
    for n, r in enumerate(rows, 1):
        i = Label.from_dict(r).dsld_id
        if i in seen:
            raise ValueError(f"row {n}: dsld_id {i} appears twice; a label set holds each id once")
        seen.add(i)

    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    passages_sha = hashlib.sha256(data).hexdigest()[:16]
    root = Path(root or LABELS_ROOT)
    d = root / name
    if d.exists():
        existing = json.loads((d / LABEL_SET_MANIFEST_NAME).read_text()).get("passages_sha")
        if existing != passages_sha:
            raise ValueError(
                f"label set {name!r} already exists with passages_sha {existing}, and these "
                f"passages hash to {passages_sha}. Changed passages need a new name."
            )
        return passages_sha

    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{name}-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        (tmp / LABEL_SET_ROWS_NAME).write_bytes(data)
        (tmp / LABEL_SET_MANIFEST_NAME).write_text(
            json.dumps(
                {"name": name, **manifest, "name": name, "passages_sha": passages_sha},
                indent=2,
                ensure_ascii=False,
            )
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        os.rename(tmp, d)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not d.exists():
            raise
        # A concurrent write won the race; the check above decides again.
        return write_label_set(name, rows, manifest, root)
    return passages_sha
