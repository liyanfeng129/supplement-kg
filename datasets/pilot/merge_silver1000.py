"""Merge validated silver shards into silver-1000.jsonl plus a manifest.

Run with .venv/bin/python datasets/pilot/merge_silver1000.py --shard-dir DIR.
No model calls. Re-checks every evidence span verbatim against the drawn
passages and refuses to write if any row is missing, misordered or ungrounded:
a reference set that is wrong is worse than one that is absent.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalise_claim_text import normalise

# Packaging/seal and child-safety lines are risks only when the label states
# no other risk text (frozen rules). Annotators applied the fallback
# correctly in nearly every case; this catches the rest mechanically.
FALLBACK_ONLY = ("safety seal", "seal is broken", "seal is damaged",
                 "seal is missing", "tamper", "do not use if the seal",
                 "reach of children", "out of reach of childen")


def is_fallback_only(evidence: str) -> bool:
    """A span that is *only* packaging or keep-out-of-reach boilerplate."""
    low = evidence.lower()
    if not any(k in low for k in FALLBACK_ONLY):
        return False
    # A block that also carries a real caution is not fallback-only.
    substantive = ("pregnan", "nursing", "medication", "physician", "doctor",
                   "consult", "adverse", "discontinue", "allerg", "interact",
                   "under the age", "medical condition", "overdose", "exceed")
    return not any(k in low for k in substantive)

INPUTS = ROOT / "data/dsld/runs/silver1000-text-v2/raw/inputs.jsonl"
DRAW_MANIFEST = ROOT / "data/dsld/runs/silver1000-text-v2/manifests/silver-1000.json"


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def merge(shard_dir, out_jsonl, out_manifest, out_sources):
    src = rows(INPUTS)
    by_id = {r["dsld_id"]: r for r in src}
    got = []
    for shard in sorted(Path(shard_dir).glob("shard-*.out.jsonl")):
        got.extend(rows(shard))

    merged = {r["dsld_id"]: r for r in got}
    missing = [i for i in by_id if i not in merged]
    extra = [i for i in merged if i not in by_id]
    if missing or extra:
        raise ValueError(f"missing {len(missing)} / unexpected {len(extra)} ids: "
                         f"{missing[:10]} {extra[:10]}")
    if len(got) != len(src):
        raise ValueError(f"{len(got)} rows from shards, expected {len(src)} (duplicates?)")

    ordered = [merged[r["dsld_id"]] for r in src]
    counts = Counter()
    problems = []
    # Shard annotators varied on whether `text` was lowercased. Gold-1000
    # lowercases claim text but keeps acronyms, so normalise here rather than
    # re-running agents: the rule is mechanical and `evidence` is untouched.
    renormalised = []
    dropped_fallback = []
    for r in ordered:
        # Drop packaging/child-safety spans when a real risk is also present.
        if len(r["risk"]) > 1:
            keep = [e for e in r["risk"] if not is_fallback_only(e["evidence"])]
            if keep and len(keep) != len(r["risk"]):
                for e in r["risk"]:
                    if e not in keep:
                        dropped_fallback.append({"dsld_id": r["dsld_id"],
                                                 "evidence": e["evidence"]})
                r["risk"] = keep
        passage = by_id[r["dsld_id"]]["passage"]
        counts["with_purpose"] += bool(r["purpose"])
        counts["with_risk"] += bool(r["risk"])
        for kind in ("purpose", "risk"):
            for i, e in enumerate(r[kind]):
                counts[f"{kind}_entries"] += 1
                fixed = normalise(e["text"])
                if fixed != e["text"]:
                    renormalised.append({"dsld_id": r["dsld_id"], "kind": kind,
                                         "index": i, "from": e["text"], "to": fixed})
                    e["text"] = fixed
                if e["evidence"] not in passage:
                    problems.append(f"{r['dsld_id']} {kind}[{i}]")
    if problems:
        raise ValueError(f"{len(problems)} ungrounded spans: {problems[:10]}")

    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered)
    Path(out_jsonl).write_text(payload)
    sources = "".join(json.dumps({"dsld_id": r["dsld_id"], "name": r["name"],
                                  "passage": r["passage"]}, ensure_ascii=False) + "\n"
                      for r in src)
    Path(out_sources).write_text(sources)

    draw = json.loads(DRAW_MANIFEST.read_text())
    gold_ids = {json.loads(l)["dsld_id"]
                for l in (ROOT / "datasets/pilot/gold-1000.jsonl").read_text().splitlines() if l.strip()}
    manifest = {
        "task": "dsld_purpose_risk_silver",
        "schema_version": 1,
        "model": "claude-sonnet-5-subagent",
        "method": "Second model-annotated reference set, disjoint from gold-1000 by "
                  "construction. Sonnet subagents read renderer-v2 passages offline and "
                  "wrote extractions directly; no extraction API calls. Not human-adjudicated.",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "renderer_version": draw["renderer_version"],
        "draw_seed": draw["seed"],
        "n_extracted": len(ordered),
        "n_unique_ids": len({r["dsld_id"] for r in ordered}),
        "n_overlap_with_gold_1000": len({r["dsld_id"] for r in ordered} & gold_ids),
        "n_with_purpose": counts["with_purpose"],
        "n_with_risk": counts["with_risk"],
        "n_purpose_entries": counts["purpose_entries"],
        "n_risk_entries": counts["risk_entries"],
        "n_ungrounded_entries": 0,
        "n_text_renormalised": len(renormalised),
        "n_fallback_risk_dropped": len(dropped_fallback),
        "n_shards": len(list(Path(shard_dir).glob("shard-*.out.jsonl"))),
        "files": {
            Path(out_jsonl).name: {"sha256": hashlib.sha256(payload.encode()).hexdigest()},
            Path(out_sources).name: {"sha256": hashlib.sha256(sources.encode()).hexdigest()},
        },
        "sampling": {k: draw[k] for k in
                     ("dsp_csv", "dump_root", "seed", "method", "disjoint_from_gold",
                      "n_frame_after_gold_removed", "n_ids_in_dump", "output_sha256")},
        "limits": "Model-annotated, not human-adjudicated ground truth. Verbatim grounding "
                  "checks quotations, not semantic completeness. Renderer v2 passages differ "
                  "from the v1 passages gold-1000 was labelled against, so cross-set "
                  "comparisons confound annotator and renderer.",
        "text_normalisation": {
            "rule": "Lowercase claim text, strip leading bullets and trailing claim asterisk, preserve acronyms. Matches gold-1000 on 4280/4281 of its entries.",
            "applied_to_entries": len(renormalised),
            "evidence_untouched": True,
        },
        "fallback_risk_rule": {
            "rule": "Packaging/seal and keep-out-of-reach spans are risks only when the label states no other risk text. Dropped where a genuine caution was also extracted for the same label.",
            "dropped_entries": len(dropped_fallback),
        },
        "dsld_ids": [r["dsld_id"] for r in ordered],
    }
    Path(out_manifest.parent / "silver-1000.normalisation-log.json").write_text(
        json.dumps({"renormalised_text": renormalised,
                    "dropped_fallback_risk": dropped_fallback},
                   indent=2, ensure_ascii=False))
    Path(out_manifest).write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-dir", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, default=ROOT / "datasets/pilot/silver-1000.jsonl")
    p.add_argument("--out-manifest", type=Path, default=ROOT / "datasets/pilot/silver-1000.manifest.json")
    p.add_argument("--out-sources", type=Path, default=ROOT / "datasets/pilot/silver-1000.sources.jsonl")
    a = p.parse_args()
    m = merge(a.shard_dir, a.out_jsonl, a.out_manifest, a.out_sources)
    print(json.dumps({k: v for k, v in m.items() if k != "dsld_ids"}, indent=2))
