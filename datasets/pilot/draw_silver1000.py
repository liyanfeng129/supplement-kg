"""Draw a second 1,000-label DSLD sample, disjoint from gold-1000.

Run with .venv/bin/python datasets/pilot/draw_silver1000.py.
No model calls. The gold set and its frozen passages are read-only.

The draw removes the gold IDs from the frame *before* sampling rather than
drawing and discarding collisions: excluding after the fact would make the
result depend on how many collisions happened to occur, so the same
(frame, seed, n) would stop redrawing the same set once gold changed.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.pipeline.render import RENDERER_VERSION, label_text
from engine.services.extraction.label_purpose_risk import label_segments
from engine.services.extraction.pilot_sample import frame
from engine.services.sources.dsld import dump_path, parse_label

SEED = 20260920


def read_ids(path):
    return {str(json.loads(line)["dsld_id"])
            for line in path.read_text().splitlines() if line.strip()}


def draw(dsp_csv, dump_root, out_dir, n, seed):
    gold_path = ROOT / "datasets/pilot/gold-1000.jsonl"
    gold_ids = read_ids(gold_path)

    f = frame(dsp_csv, dump_root)
    eligible = f.pop("eligible")
    remaining = [i for i in eligible if i not in gold_ids]
    if len(eligible) - len(remaining) != len(gold_ids):
        raise ValueError("Gold IDs are not all inside the sampling frame")

    drawn = sorted(random.Random(seed).sample(remaining, n), key=int)
    if set(drawn) & gold_ids:
        raise ValueError("Draw overlaps gold-1000")

    records, lengths, empty = [], [], 0
    raw_sha = hashlib.sha256()
    for did in drawn:
        raw_bytes = dump_path(did, dump_root).read_bytes()
        raw_sha.update(raw_bytes)
        raw = json.loads(raw_bytes)
        if str(raw["id"]) != did:
            raise ValueError(f"Raw label ID mismatch: {did}")
        label = parse_label(raw)
        passage = label_text(label)
        lengths.append(len(passage))
        empty += not passage
        records.append({"dsld_id": label.dsld_id, "name": label.full_name,
                        "passage": passage, "source_segments": label_segments(label)})

    out_dir = Path(out_dir)
    raw_dir, man_dir = out_dir / "raw", out_dir / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    (raw_dir / "inputs.jsonl").write_text(payload)

    lengths.sort()
    manifest = {
        "task": "dsld_purpose_risk_silver_sample",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "dsp_csv": str(dsp_csv),
        "dump_root": str(dump_root),
        "renderer_version": RENDERER_VERSION,
        "seed": seed,
        "n_requested": n,
        "n_drawn": len(drawn),
        "method": "Fresh seeded draw from the DSLD-only frame with the 1,000 "
                  "gold IDs removed before sampling. Disjoint from gold-1000 "
                  "by construction, not by post-hoc filtering.",
        "disjoint_from_gold": True,
        "n_frame_after_gold_removed": len(remaining),
        "gold_excluded": len(gold_ids),
        **f,
        "empty_passages": empty,
        "passage_chars": {"median": lengths[len(lengths) // 2],
                          "p95": lengths[int(len(lengths) * 0.95)],
                          "max": lengths[-1]},
        "input_sha256": {
            gold_path.name: hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            "raw_labels_concatenated_in_draw_order": raw_sha.hexdigest(),
        },
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "limits": "A mechanical draw and render. No model was called and no "
                  "purpose/risk annotation is implied by membership.",
        "dsld_ids": drawn,
    }
    (man_dir / "silver-1000.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsp-csv", type=Path, default=ROOT / "datasets/iDISK2.0/Data/Entity/DSP.csv")
    p.add_argument("--dump-root", type=Path, default=ROOT / "datasets/DSLD-full-database-JSON")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/dsld/runs/silver1000-text-v2")
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    m = draw(args.dsp_csv, args.dump_root, args.out_dir, args.n, args.seed)
    print(json.dumps({k: v for k, v in m.items() if k != "dsld_ids"}, indent=2))
