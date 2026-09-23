"""Rebuild lean extraction inputs and trace gold evidence to raw DSLD fields.

Run with .venv/bin/python datasets/pilot/prepare_gold1000_inputs.py.
No model calls; the frozen gold and its original passages are read-only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.pipeline.render import RENDERER_VERSION, label_text
from engine.services.extraction.label_purpose_risk import label_segments
from engine.services.sources.dsld import dump_path, parse_label


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(dump_root, output):
    pilot = ROOT / "datasets/pilot"
    gold_path = pilot / "gold-1000.jsonl"
    frozen_path = pilot / "gold-1000.sources.jsonl"
    gold = read_rows(gold_path)
    frozen_rows = read_rows(frozen_path)
    ids = [row["dsld_id"] for row in gold]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("Expected 1,000 unique gold labels")
    if [row["dsld_id"] for row in frozen_rows] != ids:
        raise ValueError("Frozen source IDs/order differ from gold")
    records, lost = [], []
    totals = {"purpose": 0, "risk": 0}
    raw_sha = hashlib.sha256()
    for row in gold:
        did = row["dsld_id"]
        raw_bytes = dump_path(did, dump_root).read_bytes()
        raw_sha.update(raw_bytes)
        raw = json.loads(raw_bytes)
        if raw["id"] != did:
            raise ValueError(f"Raw label ID mismatch: {did}")
        label = parse_label(raw)
        passage = label_text(label)
        segments = label_segments(label)
        trace = []
        for kind in totals:
            for index, entry in enumerate(row[kind]):
                evidence = entry["evidence"]
                paths = [path for segment in segments
                         if evidence and evidence in segment["text"]
                         for path in segment["paths"]]
                if not paths or evidence not in passage:
                    lost.append({"dsld_id": did, "kind": kind, "index": index})
                totals[kind] += 1
                trace.append({"kind": kind, "index": index, "paths": paths})
        records.append({"dsld_id": did, "name": label.full_name,
                        "passage": passage, "source_segments": segments,
                        "gold_evidence_paths": trace})
    if lost:
        raise ValueError(f"Gold evidence would be lost: {lost}")

    before = sum(len(row["passage"]) for row in frozen_rows)
    after = sum(len(row["passage"]) for row in records)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    report = {
        "renderer_version": RENDERER_VERSION,
        "labels": len(records),
        "retained_gold_claims": totals,
        "lost_gold_claims": 0,
        "frozen_passage_characters": before,
        "prepared_passage_characters": after,
        "character_reduction_fraction": round(1 - after / before, 6),
        "empty_passages": sum(not row["passage"] for row in records),
        "kept_text_fields": ["statements[].notes", "ingredientRows[].notes",
                             "ingredientRows[].description", "recursive nestedRows free text"],
        "removed_prompt_fields": ["fullName", "brandName", "productType", "physicalState",
                                  "claims", "targetGroups", "ingredient names", "statements[].type"],
        "input_sha256": {
            gold_path.name: hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            frozen_path.name: hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            "raw_labels_concatenated_in_gold_order": raw_sha.hexdigest(),
        },
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "limits": "Exact recorded evidence retention, not semantic completeness or model accuracy. "
                  "Other raw free text (e.g. contact text and otheringredients.text) remains outside "
                  "this extraction projection and is not certified irrelevant. No model was rerun.",
    }
    # Verify every label before writing any derived output.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, default=ROOT / "datasets/DSLD-full-database-JSON")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/dsld/runs/gold1000-text-v2/raw/inputs.jsonl")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dump_root, args.output), indent=2))
