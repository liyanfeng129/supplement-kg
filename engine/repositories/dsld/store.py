"""Results store: JSONL under timestamped run directories.

Follows the convention the DSLD census and pilot runs already use --
`data/dsld/runs/<TIMESTAMP>Z/`, manifest committed, `raw/*.jsonl` gitignored.
Every row carries `config_hash` and `provenance` inline, so a row is
interpretable without joining it back to a manifest, and stats are a glob into
pandas rather than a database.

SQLite is rejected here: it breaks the run-dir convention, is not
human-readable, and the agent driver would have to issue SQL in-session rather
than appending reviewable dicts. A JSONL-plus-derived-index is worse -- a stale
index is a silent wrong-answer source.

**Resume is input IDs minus IDs present in the output.** No sidecar progress
file: two files can disagree, and a checkpoint contradicting the data is worse
than none. A malformed trailing line -- a crash mid-write -- is discarded and
that record re-run, costing one record rather than corrupting a run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO_ROOT / "data" / "dsld" / "runs"

ROWS_NAME = "purpose_risk.jsonl"
MANIFEST_NAME = "purpose_risk.json"

# Cleaned, flattened label text -- the extractor's input rather than its
# output. It shares the run-dir convention so one directory can hold the
# passages a run was given and the rows it produced, side by side.
PASSAGES_NAME = "passages.jsonl"
PASSAGES_MANIFEST_NAME = "passages.json"


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_dir(rid: str | None = None, root: str | Path = RUNS_ROOT) -> Path:
    """Open a run directory, creating `raw/` and `manifests/` inside it.

    With an explicit `rid` this is how `--resume` reopens a directory, so an
    existing one is returned as-is. Without one, a *new* directory is wanted:
    run ids have second resolution, so a run started within a second of another
    -- a retry pass immediately after its parent, most likely -- would otherwise
    silently append to it and blend the two, which is exactly what keeping
    retried work attributable forbids.
    """
    root = Path(root)
    if rid is None:
        rid = base = run_id()
        for n in range(1, 100):
            if not (root / rid).exists():
                break
            rid = f"{base}-{n}"
    d = root / rid
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "manifests").mkdir(parents=True, exist_ok=True)
    return d


def rows_path(d: str | Path, name: str = ROWS_NAME) -> Path:
    """Where a run's rows live. `name` selects which artifact in the run."""
    return Path(d) / "raw" / name


def manifest_path(d: str | Path, name: str = MANIFEST_NAME) -> Path:
    """Where a run's manifest lives. `name` selects which artifact in the run."""
    return Path(d) / "manifests" / name


def read_rows(d: str | Path) -> list[dict]:
    """Read a run's rows, discarding a malformed trailing line.

    A crash mid-write leaves a partial final line. Discarding it costs the one
    record it belongs to, which resume then re-runs.
    """
    path = rows_path(d)
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


def repair(d: str | Path) -> int:
    """Rewrite the rows file without its torn trailing line, if any.

    Returns the number of rows kept. Appending after a torn line would leave
    one unparseable line in the middle of the file, where `read_rows` cannot
    safely skip it.
    """
    path = rows_path(d)
    if not path.exists():
        return 0
    rows = read_rows(d)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def write_manifest(d: str | Path, manifest: dict) -> Path:
    p = manifest_path(d)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return p


def read_manifest(d: str | Path) -> dict:
    p = manifest_path(d)
    return json.loads(p.read_text()) if p.exists() else {}


def write_passages(d: str | Path, rows: list[dict], manifest: dict) -> dict[str, Path]:
    """Save cleaned, flattened label text and the manifest describing it.

    Written whole rather than appended: cleaning a label is cheap and
    deterministic, so a interrupted run is re-cleaned rather than resumed, and
    a half-written passage file has no reason to exist. Extraction rows, which
    cost a model call each, keep their append-and-resume path.
    """
    rows_p = rows_path(d, PASSAGES_NAME)
    rows_p.parent.mkdir(parents=True, exist_ok=True)
    with rows_p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    man_p = manifest_path(d, PASSAGES_MANIFEST_NAME)
    man_p.parent.mkdir(parents=True, exist_ok=True)
    man_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"rows": rows_p, "manifest": man_p}


def load_runs(root: str | Path = RUNS_ROOT, pattern: str = "*"):
    """Glob run directories into one pandas frame.

    At ~92k rows this is tens of MB -- comparing runs needs no database.
    """
    import pandas as pd

    frames = []
    for d in sorted(Path(root).glob(pattern)):
        rows = read_rows(d) if rows_path(d).exists() else []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["run_id"] = d.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
