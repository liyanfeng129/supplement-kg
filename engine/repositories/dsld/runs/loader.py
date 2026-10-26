"""The entry to read runs: `read_manifest`, `read_ids`, `read_rows`, `todo_ids` and `read_profile`.

The store writes runs; this module only reads them, and never imports the
store. Where a run's files live is `paths.py`'s business.

Read straight from disk on every call, never cached: a run is appended to
while it goes, so a cached read would be stale.

**Done is read from the rows, never stored.** A label is done when its id has
a complete row. Only the text after the last newline can be torn -- the
`Sink` appends whole rows, each with its newline -- so that tail is skipped
unless it parses, and its label counts as todo. Nothing here repairs it:
reading must be safe while a run is live, when the tail may be a row the
writer is still writing. The `Sink` repairs the tail before it appends.

**A stored profile is checked against its run on every read.** It is only
trusted while the run's rows and manifest are the ones it was computed from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from engine.repositories.dsld.runs import paths
from engine.repositories.dsld.runs.schema import ProfiledLabel


def read_manifest(d: str | Path) -> dict:
    """The run's manifest, or `{}` if it has none."""
    p = paths.manifest_path(d)
    return json.loads(p.read_text()) if p.exists() else {}


def read_ids(d: str | Path) -> list[int]:
    """The run's ids, in the order the run does them."""
    return [
        int(json.loads(line)["dsld_id"])
        for line in paths.rows_path(d, paths.IDS_NAME).read_text().splitlines()
        if line.strip()
    ]


def read_rows(d: str | Path) -> list[dict]:
    """Read a run's complete rows, skipping a torn tail.

    The tail -- whatever follows the last newline -- is a row if it parses
    (written whole, its newline lost) and a torn write if it does not; a torn
    write costs the one label it belongs to, which resume then re-runs. A bad
    line anywhere before it raises, naming the line: it cannot be a torn
    write, so it can only be corruption.
    """
    path = paths.rows_path(d)
    if not path.exists():
        return []
    # Bytes, not text: a write torn inside a multi-byte character would make
    # the whole file fail to decode, not just its tail.
    *lines, tail = path.read_bytes().split(b"\n")
    rows = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from exc
    if tail.strip():
        try:
            rows.append(json.loads(tail))
        except ValueError:
            pass  # torn trailing write; the label is simply not done
    return rows


def todo_ids(d: str | Path) -> list[int]:
    """The run's ids without a complete row, in run order: what a resume does."""
    done = {int(r["dsld_id"]) for r in read_rows(d) if "dsld_id" in r}
    return [i for i in read_ids(d) if i not in done]


def read_profile(d: str | Path) -> tuple[dict, list[ProfiledLabel]]:
    """The run's stored profile: its manifest, and its labels in run order.

    Raises if there is none, if `profile.jsonl` exists without its manifest
    (a crashed write), and if the profile is stale: its `rows_sha`,
    `config_hash` or `passages_sha` disagree with the run's own rows and
    manifest. Whether its `classifier_version` is current is the caller's
    check, as the version belongs to the service that computes profiles.
    """
    labels_p = paths.rows_path(d, paths.PROFILE_NAME)
    manifest_p = paths.manifest_path(d, paths.PROFILE_MANIFEST_NAME)
    if not manifest_p.exists():
        if labels_p.exists():
            raise ValueError(
                f"{labels_p} exists without {manifest_p}: a crashed write. Delete it and "
                f"profile the run again."
            )
        raise FileNotFoundError(f"no profile of run {d}: build_profile() writes one")
    manifest = json.loads(manifest_p.read_text())

    # Stale: the run's rows or manifest are not the ones the profile was computed from.
    run_manifest = read_manifest(d)
    current = {
        "rows_sha": hashlib.sha256(paths.rows_path(d).read_bytes()).hexdigest()[:16],
        "config_hash": run_manifest.get("config_hash"),
        "passages_sha": run_manifest.get("passages_sha"),
    }
    for field, value in current.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"profile of run {d} is stale: its {field} is {manifest.get(field)} but the "
                f"run's is {value}. Delete both profile files and profile the run again."
            )

    labels = [
        ProfiledLabel(
            dsld_id=int(r["dsld_id"]),
            group=r["group"],
            purpose_types=list(r["purpose_types"]),
            risk_types=list(r["risk_types"]),
        )
        for r in (json.loads(line) for line in labels_p.read_text().splitlines() if line.strip())
    ]
    return manifest, labels
