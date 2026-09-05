"""Whole-database DSLD harvest, with an iDISK intersection.

Fetching label-by-label costs one request per DSLD ID -- ~157k requests, about
nine hours at the client's self-imposed throttle -- and ODS publishes no bulk
export. `search-filter` returns full label documents 1,000 at a time instead,
which brings the same corpus down to a few hundred requests.

The catch is that paging stops at `MAX_RESULT_WINDOW`, so the database is
harvested as disjoint slices (see `DSLDClient.plan_slices`) and reassembled
here. Harvesting everything and filtering afterwards is cheaper than looking
up the iDISK IDs individually, and it leaves a reusable full-corpus snapshot
behind rather than only the rows this project happens to want today.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .dsld import DSLD_API, DSLDClient, Label, parse_dsld_ids, parse_label


def idisk_dsld_ids(dsp_csv: str | Path) -> set[str]:
    """Every DSLD ID referenced by an iDISK DSP row."""
    import pandas as pd

    df = pd.read_csv(dsp_csv, dtype=str)
    return {i for cell in df.get("DSLD_ID", []) for i in parse_dsld_ids(cell)}


def harvest(
    out_dir: str | Path,
    dsp_csv: str | Path | None = None,
    client: DSLDClient | None = None,
    run_name: str | None = None,
) -> dict[str, Path]:
    """Harvest all DSLD labels, and mark which ones iDISK points at.

    Writes the full corpus as JSONL plus a manifest. When `dsp_csv` is given,
    each label is tagged `in_idisk` and the corpus is also split into the
    iDISK subset and its complement -- the labels DSLD has gained since
    iDISK's snapshot, which is where a refresh has anything to add. The
    manifest records which requested IDs DSLD has no label for.

    The split is possible because the census *is* the ID discovery step:
    `search-filter` returns each hit's DSLD ID, so the new labels are simply
    the ones iDISK never mentions. Those IDs are what a later
    `fetch_concurrent()` sweep needs, since nothing else enumerates them.
    """
    client = client or DSLDClient()
    started = datetime.now(timezone.utc)
    run_name = run_name or f"dsld-full-{started:%Y%m%dT%H%M%SZ}"

    wanted = idisk_dsld_ids(dsp_csv) if dsp_csv else set()

    out_dir = Path(out_dir)
    raw_dir, man_dir = out_dir / "raw", out_dir / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    slices = client.plan_slices()
    expected_total = sum(s["expected"] for s in slices)
    reported_total = client.search_count()

    all_path = raw_dir / f"{run_name}.jsonl"
    sub_path = raw_dir / f"{run_name}-idisk.jsonl"
    new_path = raw_dir / f"{run_name}-new.jsonl"
    ids_path = raw_dir / f"{run_name}-new-ids.txt"

    seen: set[int] = set()
    matched: set[str] = set()
    n_all = n_sub = n_new = 0

    with all_path.open("w") as fa, sub_path.open("w") as fs, \
            new_path.open("w") as fn, ids_path.open("w") as fi:
        for doc in client.iter_all(slices):
            if doc["id"] in seen:
                continue
            seen.add(doc["id"])
            label: Label = parse_label(doc)
            in_idisk = str(label.dsld_id) in wanted
            # Tag every row, so a consumer loading the full corpus can split
            # it without re-reading the iDISK CSV.
            row = json.dumps(asdict(label) | {"in_idisk": in_idisk})
            fa.write(row + "\n")
            n_all += 1
            if in_idisk:
                fs.write(row + "\n")
                matched.add(str(label.dsld_id))
                n_sub += 1
            elif wanted:
                fn.write(row + "\n")
                # A bare ID list is what feeds a follow-up statement sweep.
                fi.write(f"{label.dsld_id}\n")
                n_new += 1

    if not wanted:
        for path in (sub_path, new_path, ids_path):
            path.unlink()

    manifest = {
        "source": "dsld",
        "mode": "bulk",
        "api": DSLD_API,
        "executed_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        # These three should agree; a gap means a slice silently truncated.
        "reported_total": reported_total,
        "expected_total": expected_total,
        "fetched": n_all,
        "complete": n_all == reported_total == expected_total,
        "slices": slices,
        "idisk_requested": len(wanted),
        "idisk_matched": n_sub,
        # Labels DSLD has that iDISK's snapshot predates.
        "new_since_idisk": n_new,
        "idisk_missing": sorted(wanted - matched),
    }
    man_path = man_dir / f"{run_name}.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    paths = {"raw": all_path, "manifest": man_path}
    if wanted:
        paths |= {"idisk": sub_path, "new": new_path, "new_ids": ids_path}
    return paths
