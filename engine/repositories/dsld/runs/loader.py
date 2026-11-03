"""The entry to read runs: `read_manifest`, `read_ids`, `read_rows`, `todo_ids`, `read_profile`, `read_pilot`, `read_evaluation_set` and `read_human_verdicts`.

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
So is a stored sample, against the profile and rows it was drawn from; its
read joins each sampled label with its claims' text (from the rows) and type
(from the profile).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from engine.repositories.dsld.runs import paths
from engine.repositories.dsld.runs.schema import (
    Adjudication,
    Claim,
    ClaimVerdict,
    HumanVerdict,
    MissedClaim,
    ProfiledLabel,
    SampledCase,
    SampledLabel,
)


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


def read_pilot(d: str | Path) -> list[SampledCase]:
    """The run's pilot, each label joined with its claims, in stored order.

    Raises if there is none, if `pilot.jsonl` exists without its manifest (a
    crashed write), and if it is stale: the run's profile or rows are not the
    ones it was drawn from. The pilot's human part is `c.label.in_human`.
    """
    lines_p = paths.eva_path(d, paths.PILOT_NAME)
    manifest_p = paths.eva_manifest_path(d, paths.PILOT_MANIFEST_NAME)
    if not manifest_p.exists():
        if lines_p.exists():
            raise ValueError(f"{lines_p} exists without {manifest_p}: a crashed write. Delete it and draw again.")
        raise FileNotFoundError(f"no pilot of run {d}: build_pilot() draws one")
    manifest = json.loads(manifest_p.read_text())

    # Stale: the profile or rows are not the ones the pilot was drawn from.
    current = {
        "profile_sha": hashlib.sha256(paths.rows_path(d, paths.PROFILE_NAME).read_bytes()).hexdigest()[:16],
        "rows_sha": hashlib.sha256(paths.rows_path(d).read_bytes()).hexdigest()[:16],
    }
    for field, value in current.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"pilot of run {d} is stale: its {field} is {manifest.get(field)} but the run's is "
                f"{value}. Its strata may no longer hold."
            )

    # Join each label with its claims: text from the row, type from the profile, by index.
    _, profiled = read_profile(d)
    types = {p.dsld_id: p for p in profiled}
    rows = {int(r["dsld_id"]): r for r in read_rows(d)}
    cases = []
    for line in lines_p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        dsld_id = int(r["dsld_id"])
        label = SampledLabel(r["case_id"], dsld_id, (r["group"], r["rarest"]), r["rank"], r["in_human"])
        cases.append(
            SampledCase(
                label,
                [
                    Claim(f"claim:{dsld_id}:purpose:{i}", item["text"], types[dsld_id].purpose_types[i])
                    for i, item in enumerate(rows[dsld_id]["purpose"])
                ],
                [
                    Claim(f"claim:{dsld_id}:risk:{i}", item["text"], types[dsld_id].risk_types[i])
                    for i, item in enumerate(rows[dsld_id]["risk"])
                ],
            )
        )
    return cases


def read_evaluation_set(d: str | Path) -> list[SampledCase]:
    """The run's evaluation set (the large set), each label joined with its claims, in stored order.

    Raises as `read_pilot` does, and also if the pilot it was drawn after has
    changed. The human set is the cases with `c.label.in_human`.
    """
    lines_p = paths.eva_path(d, paths.EVALUATION_SET_NAME)
    manifest_p = paths.eva_manifest_path(d, paths.EVALUATION_SET_MANIFEST_NAME)
    if not manifest_p.exists():
        if lines_p.exists():
            raise ValueError(f"{lines_p} exists without {manifest_p}: a crashed write. Delete it and draw again.")
        raise FileNotFoundError(f"no evaluation set of run {d}: build_evaluation_set() draws one")
    manifest = json.loads(manifest_p.read_text())

    # Stale: the profile, rows or pilot are not the ones the set was drawn from.
    pilot_p = paths.eva_path(d, paths.PILOT_NAME)
    current = {
        "profile_sha": hashlib.sha256(paths.rows_path(d, paths.PROFILE_NAME).read_bytes()).hexdigest()[:16],
        "rows_sha": hashlib.sha256(paths.rows_path(d).read_bytes()).hexdigest()[:16],
        "pilot_sha": hashlib.sha256(pilot_p.read_bytes()).hexdigest()[:16] if pilot_p.exists() else None,
    }
    for field, value in current.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"evaluation set of run {d} is stale: its {field} is {manifest.get(field)} but the "
                f"run's is {value}. Its strata or weights may no longer hold."
            )

    # Join each label with its claims: text from the row, type from the profile, by index.
    _, profiled = read_profile(d)
    types = {p.dsld_id: p for p in profiled}
    rows = {int(r["dsld_id"]): r for r in read_rows(d)}
    cases = []
    for line in lines_p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        dsld_id = int(r["dsld_id"])
        label = SampledLabel(r["case_id"], dsld_id, (r["group"], r["rarest"]), r["rank"], r["in_human"])
        cases.append(
            SampledCase(
                label,
                [
                    Claim(f"claim:{dsld_id}:purpose:{i}", item["text"], types[dsld_id].purpose_types[i])
                    for i, item in enumerate(rows[dsld_id]["purpose"])
                ],
                [
                    Claim(f"claim:{dsld_id}:risk:{i}", item["text"], types[dsld_id].risk_types[i])
                    for i, item in enumerate(rows[dsld_id]["risk"])
                ],
            )
        )
    return cases


def read_human_verdicts(d: str | Path) -> dict[str, HumanVerdict]:
    """The human's verdicts on the run's pilot, by `case_id`: the last save of each label.

    Empty when none is saved. A torn tail is skipped, as `read_rows` skips
    one; a bad line before it raises, naming the line.
    """
    path = paths.eva_path(d, paths.HUMAN_PILOT_NAME)
    if not path.exists():
        return {}
    *lines, tail = path.read_bytes().split(b"\n")
    records = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from exc
    if tail.strip():
        try:
            records.append(json.loads(tail))
        except ValueError:
            pass  # torn trailing write; that save is lost

    # Later lines replace earlier ones: the last save of a label is its verdict.
    verdicts = {}
    for r in records:
        verdicts[r["case_id"]] = HumanVerdict(
            case_id=r["case_id"],
            dsld_id=int(r["dsld_id"]),
            claims=[ClaimVerdict(**c) for c in r["claims"]],
            missed=[MissedClaim(**m) for m in r["missed"]],
            opened_at=r["opened_at"],
            saved_at=r["saved_at"],
            active_seconds=int(r["active_seconds"]),
        )
    return verdicts


def read_adjudications(d: str | Path) -> dict[tuple[str, str], Adjudication]:
    """The human's decisions after the reveal, by `(case_id, item)`: the last decision on each item.

    Empty when none is stored. A torn tail is skipped, as `read_human_verdicts`
    skips one; a bad line before it raises, naming the line.
    """
    path = paths.eva_path(d, paths.ADJUDICATION_PILOT_NAME)
    if not path.exists():
        return {}
    *lines, tail = path.read_bytes().split(b"\n")
    decisions = {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from exc
        decisions[(r["case_id"], r["item"])] = Adjudication(**r)
    if tail.strip():
        try:
            r = json.loads(tail)
        except ValueError:
            return decisions  # torn trailing write; that decision is lost
        decisions[(r["case_id"], r["item"])] = Adjudication(**r)
    return decisions
