"""The entry to save runs: `new_run`, `repair`, `Sink`, `write_profile`, `save_pilot`, `save_evaluation_set` and `save_human_verdict`, as JSONL under `paths.RUNS_ROOT`.

**A run is created whole, then only appended to.** `new_run` writes
a run directory with its ids, in run order, and its manifest together, so a
crash leaves the whole run or none of it. Results are appended to the run's
rows file by a `Sink`. Every row
carries `config_hash` inline, so a row is interpretable without joining it
back to a manifest.

**Done is read from the results, never stored.** A label is done when its id
has a complete row; the loader reads that, as this module only writes. There
is no status field to update: JSONL cannot change a line in place, and a
status beside a result is two facts that a crash can leave disagreeing. A
torn trailing line -- a crash mid-write -- costs that one label, which resume
re-runs; the `Sink` repairs it before appending.

**A profile is written once, beside the finished run.** `write_profile` saves
each label's group and claim types and the profile's manifest, never over an
existing one: re-profiling means deleting both files by hand.

**An evaluation sample is written once, in the run's `eva/`.** The pilot and
the evaluation set each go in their own file and manifest, never over an
existing one; each manifest pins the profile and rows it was drawn from.

**The human's verdicts are appended, one line per save.** A label saved again
gets a new line, and the loader takes the last one per label; every earlier
save stays as history. A torn last line is cut before the next append.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from engine import config
from engine.repositories.dsld.runs import paths
from engine.repositories.dsld.runs.schema import Adjudication, HumanVerdict, ProfiledLabel, SampledLabel

# Used when `configs/engine.toml` has no `[store]` value.
_FALLBACK_FLUSH_EVERY = 64
_FALLBACK_FLUSH_SECONDS = 30.0

# How far back `repair` reads at a time while looking for the last newline.
_REPAIR_CHUNK = 1 << 16


def repair(d: str | Path) -> None:
    """Make the rows file end on a whole row, so it can be appended to.

    Only the tail -- whatever follows the last newline -- is looked at: the
    `Sink` appends whole rows, so nothing before it can be torn. A tail that
    parses is a row whose newline was lost, and gets one; anything else is a
    torn write, and is cut off. This is the rule `loader.read_rows` reads by,
    so a label the loader counted done stays done.

    Appending after a torn tail would glue the next row onto it, leaving a
    broken line in the middle of the file, where the loader raises. A bad line
    there already is not this function's to fix: it is corruption, and the
    loader reports it.
    """
    path = paths.rows_path(d)
    if not path.exists():
        return
    with path.open("rb+") as fh:
        end = fh.seek(0, os.SEEK_END)
        # Walk back from the end to just after the last newline.
        start = end
        while start > 0:
            step = min(_REPAIR_CHUNK, start)
            fh.seek(start - step)
            nl = fh.read(step).rfind(b"\n")
            if nl != -1:
                start = start - step + nl + 1
                break
            start -= step
        if start == end:
            return  # ends on a whole row, or is empty
        fh.seek(start)
        tail = fh.read()
        try:
            json.loads(tail)
        except ValueError:
            fh.truncate(start)
        else:
            fh.write(b"\n")
        fh.flush()
        os.fsync(fh.fileno())


class Sink:
    """Where a streaming run's rows go: buffered, then appended in batches.

    `add` costs a list append and two comparisons, so it is safe on the loop
    that feeds the model. A row reaches disk when `flush_every` rows have
    queued up or `flush_seconds` have passed since the last flush -- whichever
    comes first -- and a crash loses only what is still in the buffer, which
    resume re-runs.

    `flush` appends and fsyncs; it never rewrites, so a crash mid-flush can
    tear only the last line, which the loader skips. Opening a sink on an
    existing run repairs that line first, for the reason `repair` exists.

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
        self.path = paths.rows_path(run)
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
    root = Path(root or paths.RUNS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".new-run-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        (tmp / "raw").mkdir()
        (tmp / "manifests").mkdir()
        with paths.rows_path(tmp, paths.IDS_NAME).open("w") as fh:
            for i in ids:
                fh.write(json.dumps({"dsld_id": int(i)}) + "\n")
        paths.manifest_path(tmp).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        stamp = paths.run_id()
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


def write_profile(d: str | Path, labels: list[ProfiledLabel], manifest: dict) -> None:
    """Save a run's profile: `raw/profile.jsonl`, then `manifests/profile.json`, all or nothing.

    Refuses if either file exists: a profile is never overwritten, and one
    file without the other is a crashed write to clean up by hand. The
    manifest gets `rows_sha`, the first 16 hex digits of the sha256 of the
    run's rows file, so a read can tell rows replaced in place.

    Each file is written to a hidden temporary name in its own folder, fsynced
    and renamed into place, the labels first and the manifest last: the
    manifest is the commit marker, so a profile exists only if it does.
    """
    labels_p = paths.rows_path(d, paths.PROFILE_NAME)
    manifest_p = paths.manifest_path(d, paths.PROFILE_MANIFEST_NAME)
    for p in (labels_p, manifest_p):
        if p.exists():
            raise FileExistsError(
                f"{p} exists: a profile is never overwritten. Delete both profile files "
                f"(and the run's eva/, drawn from them) by hand to re-profile."
            )
    rows_sha = hashlib.sha256(paths.rows_path(d).read_bytes()).hexdigest()[:16]
    manifest = {**manifest, "rows_sha": rows_sha}

    # Labels first, manifest last, each through a temporary name in its own folder.
    for target, data in (
        (labels_p, "".join(json.dumps(asdict(l), ensure_ascii=False) + "\n" for l in labels)),
        (manifest_p, json.dumps(manifest, indent=2, ensure_ascii=False)),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}-{uuid.uuid4().hex}"
        try:
            with tmp.open("w") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)


def save_pilot(d: str | Path, pilot: list[SampledLabel], manifest: dict) -> None:
    """Save a run's pilot: `eva/pilot.jsonl`, then `eva/manifests/pilot.json`, all or nothing.

    Refuses if either file exists. A line is a `SampledLabel` with its
    stratum as two flat fields, `group` and `rarest`. The manifest gets
    `profile_sha` and `rows_sha`, the first 16 hex digits of the sha256 of the
    run's profile and rows files, so a read can tell either changed. Written
    as `write_profile` writes: lines first, manifest last as the commit marker.
    """
    lines_p = paths.eva_path(d, paths.PILOT_NAME)
    manifest_p = paths.eva_manifest_path(d, paths.PILOT_MANIFEST_NAME)
    for p in (lines_p, manifest_p):
        if p.exists():
            raise FileExistsError(
                f"{p} exists: a pilot is never overwritten. Delete both pilot files (and the "
                f"evaluation set, drawn after it) by hand to draw it again."
            )
    manifest = {
        **manifest,
        "profile_sha": hashlib.sha256(paths.rows_path(d, paths.PROFILE_NAME).read_bytes()).hexdigest()[:16],
        "rows_sha": hashlib.sha256(paths.rows_path(d).read_bytes()).hexdigest()[:16],
    }
    lines = "".join(
        json.dumps(
            {
                "case_id": l.case_id,
                "dsld_id": l.dsld_id,
                "group": l.stratum[0],
                "rarest": l.stratum[1],
                "rank": l.rank,
                "in_human": l.in_human,
            }
        )
        + "\n"
        for l in pilot
    )

    # Lines first, manifest last, each through a temporary name in its own folder.
    for target, data in ((lines_p, lines), (manifest_p, json.dumps(manifest, indent=2, ensure_ascii=False))):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}-{uuid.uuid4().hex}"
        try:
            with tmp.open("w") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)


def save_evaluation_set(d: str | Path, large: list[SampledLabel], manifest: dict) -> None:
    """Save a run's evaluation set: `eva/evaluation_set.jsonl`, then its manifest, all or nothing.

    As `save_pilot`, and the manifest also gets `pilot_sha`, of
    `eva/pilot.jsonl`, the pilot it was drawn after. The human set is the
    lines with `in_human`.
    """
    lines_p = paths.eva_path(d, paths.EVALUATION_SET_NAME)
    manifest_p = paths.eva_manifest_path(d, paths.EVALUATION_SET_MANIFEST_NAME)
    for p in (lines_p, manifest_p):
        if p.exists():
            raise FileExistsError(
                f"{p} exists: an evaluation set is never overwritten. Delete both of its files "
                f"(and every later file in eva/, joined on it) by hand to draw it again."
            )
    manifest = {
        **manifest,
        "profile_sha": hashlib.sha256(paths.rows_path(d, paths.PROFILE_NAME).read_bytes()).hexdigest()[:16],
        "rows_sha": hashlib.sha256(paths.rows_path(d).read_bytes()).hexdigest()[:16],
        "pilot_sha": hashlib.sha256(paths.eva_path(d, paths.PILOT_NAME).read_bytes()).hexdigest()[:16],
    }
    lines = "".join(
        json.dumps(
            {
                "case_id": l.case_id,
                "dsld_id": l.dsld_id,
                "group": l.stratum[0],
                "rarest": l.stratum[1],
                "rank": l.rank,
                "in_human": l.in_human,
            }
        )
        + "\n"
        for l in large
    )

    # Lines first, manifest last, each through a temporary name in its own folder.
    for target, data in ((lines_p, lines), (manifest_p, json.dumps(manifest, indent=2, ensure_ascii=False))):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}-{uuid.uuid4().hex}"
        try:
            with tmp.open("w") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)


def save_human_verdict(d: str | Path, verdict: HumanVerdict) -> None:
    """Append one human verdict to the run's `eva/human_pilot.jsonl`.

    The line is the verdict as a dict (`asdict`), written whole and fsynced. A
    tail without its newline is mended first, by `repair`'s rule: one that
    parses gets its newline, anything else was a torn write and is cut off,
    so the new line is never glued onto it.
    """
    path = paths.eva_path(d, paths.HUMAN_PILOT_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Mend a torn tail: the file is small, so it is read whole.
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            start = data.rfind(b"\n") + 1
            try:
                json.loads(data[start:])
            except ValueError:
                data = data[:start]
            else:
                data += b"\n"
            with path.open("wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())

    with path.open("a") as fh:
        fh.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def save_adjudication(d: str | Path, adjudication: Adjudication) -> None:
    """Append one decision to the run's `eva/adjudication_pilot.jsonl`.

    The line is the decision as a dict (`asdict`), written whole and fsynced. A
    tail without its newline is mended first, by `repair`'s rule: one that
    parses gets its newline, anything else was a torn write and is cut off,
    so the new line is never glued onto it.
    """
    path = paths.eva_path(d, paths.ADJUDICATION_PILOT_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Mend a torn tail: the file is small, so it is read whole.
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            start = data.rfind(b"\n") + 1
            try:
                json.loads(data[start:])
            except ValueError:
                data = data[:start]
            else:
                data += b"\n"
            with path.open("wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())

    with path.open("a") as fh:
        fh.write(json.dumps(asdict(adjudication), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
