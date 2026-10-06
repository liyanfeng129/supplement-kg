"""Where the full-database dump lives. Locations only: nothing here reads or writes.

The dump is one JSON file per product, bucketed a thousand ids to a directory:
`<DUMP_ROOT>/000000-000999/709.json`. A test redirects the dump by patching
`DUMP_ROOT` here, and nowhere else: it is read at call time, never bound as a
default argument.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

DUMP_ROOT = REPO_ROOT / "datasets" / "DSLD-full-database-JSON"

# Dump files are bucketed a thousand ids to a directory.
DUMP_BUCKET_SIZE = 1000


def dump_path(dsld_id: int | str, root: str | Path | None = None) -> Path:
    """Where `dsld_id` lives in a dump. No existence check."""
    i = int(dsld_id)
    lo = (i // DUMP_BUCKET_SIZE) * DUMP_BUCKET_SIZE
    bucket = f"{lo:06d}-{lo + DUMP_BUCKET_SIZE - 1:06d}"
    return Path(root or DUMP_ROOT) / bucket / f"{i}.json"
