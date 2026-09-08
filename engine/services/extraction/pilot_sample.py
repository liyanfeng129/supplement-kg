"""Draw the DSLD-only sampling frame for the Purpose/Risk pilot.

iDISK 2.0 leaves `Purpose` and `Risk` empty on every DSLD-only DSP row
because it takes both fields from LNHPD alone (docs/dsld-purpose-risk-
findings.md section 1). Recovering them means running an extractor over DSLD
label text, and that starts with deciding which labels to run it over.

Two things about the frame are easy to get wrong:

- **Grain.** `DSLD_ID` cells are pipe-delimited, so one DSP row can name
  several labels and one label can be named by several rows. Sampling rows
  would draw a biased, ambiguous set -- the pipeline iterates labels, and a
  `Label` is keyed on a DSLD ID, so the sample is drawn at DSLD-ID grain from
  the deduped union.
- **Coverage.** iDISK may reference IDs the dump snapshot lacks. Those are
  excluded from the frame (they have no text to extract) but counted in the
  manifest: a shortfall of a few dozen is noise, thousands would change what
  extending to the whole corpus means.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.services.sources.dsld import dump_ids, parse_dsld_ids

DEFAULT_SEED = 20260908


def _has_nhp(cell: object) -> bool:
    """Whether a DSP row carries an LNHPD product ID."""
    if cell is None:
        return False
    text = str(cell).strip()
    return bool(text) and text.lower() != "nan"


def frame(dsp_csv: str | Path, dump_root: str | Path) -> dict:
    """Build the DSLD-only sampling frame and measure dump coverage.

    Returns the eligible IDs alongside the counts a manifest needs. The
    `both` rows -- products carrying curated LNHPD Purpose/Risk *and* a DSLD
    label -- are counted but excluded: they are held back as an independent
    validation set and must not be spent on prompt iteration.
    """
    df = pd.read_csv(dsp_csv, dtype=str)

    dsld_ids = df["DSLD_ID"].map(parse_dsld_ids)
    has_dsld = dsld_ids.map(bool)
    has_nhp = df["NHP_ID"].map(_has_nhp)

    only_rows = has_dsld & ~has_nhp
    both_rows = has_dsld & has_nhp

    wanted: set[str] = set()
    for ids in dsld_ids[only_rows]:
        wanted.update(ids)

    available = dump_ids(dump_root)
    eligible = sorted(wanted & available, key=int)
    missing = sorted(wanted - available, key=int)

    return {
        "eligible": eligible,
        "n_dsp_rows": int(len(df)),
        "n_dsld_only_rows": int(only_rows.sum()),
        "n_both_rows_held_back": int(both_rows.sum()),
        "n_ids_referenced": len(wanted),
        "n_ids_in_dump": len(eligible),
        "n_ids_missing_from_dump": len(missing),
        # Listed in full: small enough to read, and the only record of which
        # labels the pilot could not have covered.
        "ids_missing_from_dump": missing,
    }


def sample(
    dsp_csv: str | Path,
    dump_root: str | Path,
    n: int = 1000,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], dict]:
    """Draw `n` DSLD IDs from the DSLD-only frame. Returns (ids, manifest)."""
    f = frame(dsp_csv, dump_root)
    eligible = f.pop("eligible")

    # Sorted input plus an explicit seed: the same (frame, seed, n) redraws
    # the same sample, so a rerun compares against the earlier one rather
    # than silently scoring a different set of labels.
    rng = random.Random(seed)
    drawn = sorted(rng.sample(eligible, min(n, len(eligible))), key=int)

    manifest = {
        "task": "dsld_purpose_risk_pilot_sample",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "dsp_csv": str(dsp_csv),
        "dump_root": str(dump_root),
        "seed": seed,
        "n_requested": n,
        "n_drawn": len(drawn),
        **f,
        "dsld_ids": drawn,
    }
    return drawn, manifest


def build_pilot(
    dsp_csv: str | Path,
    dump_root: str | Path,
    out_dir: str | Path,
    n: int = 1000,
    seed: int = DEFAULT_SEED,
    run_name: str = "pilot",
) -> dict[str, Path]:
    """Draw a sample, flatten each label to one passage, write it out.

    This step is entirely mechanical -- no model is called. The JSONL it
    writes is the extractor's input, so a prompt can be re-run against a
    frozen set of passages without redrawing or re-reading the dump.
    """
    from engine.services.extraction.label_purpose_risk import label_text
    from engine.services.sources.dsld import load_label

    ids, manifest = sample(dsp_csv, dump_root, n=n, seed=seed)

    out_dir = Path(out_dir)
    raw_dir, man_dir = out_dir / "raw", out_dir / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{run_name}-{len(ids)}.jsonl"
    lengths: list[int] = []
    with raw_path.open("w") as fh:
        for did in ids:
            label = load_label(did, dump_root)
            passage = label_text(label)
            lengths.append(len(passage))
            fh.write(
                json.dumps(
                    {
                        "dsld_id": label.dsld_id,
                        "name": label.full_name,
                        "passage": passage,
                    }
                )
                + "\n"
            )

    lengths.sort()
    manifest["passage_chars"] = {
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int(len(lengths) * 0.95)],
        "max": lengths[-1],
    }
    man_path = man_dir / f"{run_name}-{len(ids)}.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    return {"raw": raw_path, "manifest": man_path}
