"""Where runs live. Locations only: nothing here reads or writes.

A run is a directory under `RUNS_ROOT`, `<set name>-<YYMMDDHHMM>-<n>/`, holding
`raw/ids.jsonl` (its ids, in run order), `raw/purpose_risk.jsonl` (its rows)
and `manifests/purpose_risk.json`, and once profiled `raw/profile.jsonl` and
`manifests/profile.json`. Its evaluation lives in its `eva/` folder: the
sampled labels in `eva/pilot.jsonl` and `eva/evaluation_set.jsonl`, their
manifests in `eva/manifests/`. A test redirects runs by patching
`RUNS_ROOT` here, and nowhere else: it is read at call time, never bound as a
default argument.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

RUNS_ROOT = REPO_ROOT / "data" / "dsld" / "runs"

IDS_NAME = "ids.jsonl"
ROWS_NAME = "purpose_risk.jsonl"
MANIFEST_NAME = "purpose_risk.json"

# A finished run's profile: `raw/profile.jsonl` and `manifests/profile.json`.
PROFILE_NAME = "profile.jsonl"
PROFILE_MANIFEST_NAME = "profile.json"

# The run's evaluation sample: `eva/<name>.jsonl` and `eva/manifests/<name>.json`.
PILOT_NAME = "pilot.jsonl"
PILOT_MANIFEST_NAME = "pilot.json"
EVALUATION_SET_NAME = "evaluation_set.jsonl"
EVALUATION_SET_MANIFEST_NAME = "evaluation_set.json"


def run_id() -> str:
    """The current UTC minute, `YYMMDDHHMM`: the time part of a run's name."""
    return datetime.now(timezone.utc).strftime("%y%m%d%H%M")


def rows_path(d: str | Path, name: str = ROWS_NAME) -> Path:
    """Where a run's rows live. `name` selects which artifact in the run."""
    return Path(d) / "raw" / name


def manifest_path(d: str | Path, name: str = MANIFEST_NAME) -> Path:
    """Where a run's manifest lives. `name` selects which artifact in the run."""
    return Path(d) / "manifests" / name


def eva_path(d: str | Path, name: str) -> Path:
    """Where one of a run's evaluation files lives, in its `eva/` folder."""
    return Path(d) / "eva" / name


def eva_manifest_path(d: str | Path, name: str) -> Path:
    """Where the manifest of one of a run's evaluation files lives."""
    return Path(d) / "eva" / "manifests" / name
