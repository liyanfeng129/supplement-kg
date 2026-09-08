"""Build the gold set the pilot is scored against.

The plan is to run a local model over the whole DSLD-only corpus. Without a
reference set, the first local run is a few hundred outputs with no way to
tell a good one from a plausible one -- so a subset goes through the frontier
model first, evidence spans kept, and every later run is scored against it.

Two things keep this honest:

- **It is committed outside `data/`.** The tree is gitignored and gets
  cleared; the gold set costs money to regenerate and every future comparison
  depends on it, so it must not be destroyable by clearing a cache.
- **The 208 `both` rows are not in it.** Those products carry curated LNHPD
  Purpose/Risk *and* a DSLD label, which makes them the only independent
  check available. Spending them on prompt iteration would leave nothing to
  validate against. They are held back deliberately -- see `pilot_sample`.

Note the gold set is a *reference*, not ground truth: it is one model's
reading of the label, auditable because the evidence span is kept, not a
human adjudication.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from engine.services.extraction.label_purpose_risk import (
    MODEL,
    Extraction,
    extract,
    _client,
)
from engine.services.sources.dsld import load_label

# Committed, so deliberately outside the gitignored `data/` tree.
DEFAULT_OUT = Path("datasets/pilot/gold-100.jsonl")


def _pilot_ids(pilot_jsonl: str | Path, n: int) -> list[str]:
    """First `n` DSLD IDs of the pilot file.

    The pilot draw was already randomised and is written in a fixed order, so
    taking a prefix is a random subset -- and one that stays the same on a
    rerun, which a second `random.sample` here would not.
    """
    ids = []
    with Path(pilot_jsonl).open() as fh:
        for line in fh:
            if len(ids) >= n:
                break
            ids.append(str(json.loads(line)["dsld_id"]))
    return ids


def build(
    pilot_jsonl: str | Path,
    dump_root: str | Path,
    out_path: str | Path = DEFAULT_OUT,
    n: int = 100,
    model: str = MODEL,
    manifest_path: str | Path | None = None,
) -> dict[str, Path]:
    """Run the frontier extractor over `n` pilot labels and write the gold set.

    Writes incrementally: a run interrupted partway leaves the labels it
    already paid for on disk rather than losing them.
    """
    client = _client()
    ids = _pilot_ids(pilot_jsonl, n)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: list[Extraction] = []
    errors: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()

    with out_path.open("w") as fh:
        for did in ids:
            try:
                e = extract(load_label(did, dump_root), client=client, model=model)
            except Exception as exc:  # noqa: BLE001 - one bad label must not sink the run
                errors.append({"dsld_id": did, "error": repr(exc)})
                continue
            done.append(e)
            fh.write(json.dumps(asdict(e)) + "\n")
            fh.flush()

    ungrounded = sum(
        1 for e in done for x in (*e.purpose, *e.risk) if not x.grounded
    )
    manifest = {
        "task": "dsld_purpose_risk_gold",
        "model": model,
        "executed_at": started,
        "pilot_jsonl": str(pilot_jsonl),
        "dump_root": str(dump_root),
        "n_requested": n,
        "n_extracted": len(done),
        "n_with_purpose": sum(1 for e in done if e.purpose),
        "n_with_risk": sum(1 for e in done if e.risk),
        "n_purpose_entries": sum(len(e.purpose) for e in done),
        "n_risk_entries": sum(len(e.risk) for e in done),
        # Expected to be 0 for a frontier model. Anything else means the
        # prompt or the passage rendering needs a look before this set is
        # trusted as a reference.
        "n_ungrounded_entries": ungrounded,
        "errors": errors,
        "dsld_ids": ids,
    }

    man_path = Path(manifest_path) if manifest_path else out_path.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2))
    return {"gold": out_path, "manifest": man_path}
