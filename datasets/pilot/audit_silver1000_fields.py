"""Trace silver evidence through frozen v2 segments to original DSLD fields.

Run with Python 3 from any directory. Uses only the standard library and local
files; writes a summary JSON, per-claim JSONL, and per-location CSV beside silver.
All input hashes, segments, and quotations are checked before writing outputs.
"""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "datasets/pilot"
RUN = ROOT / "data/dsld/runs/silver1000-text-v2"
KINDS = ("purpose", "risk")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(text):
    return " ".join(text.split())


def strings(value, path=""):
    """Visit every raw string leaf, including fields outside the renderer."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from strings(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from strings(child, f"{path}[{index}]")


def family(path):
    return re.sub(r"\[\d+\]", "[]", path)


def occurrences(text, quote):
    start = text.find(quote)
    while start >= 0:
        yield start
        start = text.find(quote, start + 1)


def audit():
    silver_path = PILOT / "silver-1000.jsonl"
    sources_path = PILOT / "silver-1000.sources.jsonl"
    inputs_path = RUN / "raw/inputs.jsonl"
    manifest_path = PILOT / "silver-1000.manifest.json"
    draw_path = RUN / "manifests/silver-1000.json"
    manifest = json.loads(manifest_path.read_text())
    draw = json.loads(draw_path.read_text())
    for name, info in manifest["files"].items():
        require(digest(PILOT / name) == info["sha256"], f"Changed frozen file: {name}")
    require(digest(inputs_path) == draw["output_sha256"], "Changed prepared inputs")
    silver, sources, inputs = rows(silver_path), rows(sources_path), rows(inputs_path)
    ids = [row["dsld_id"] for row in silver]
    require(len(ids) == len(set(ids)) == 1000, "Expected 1,000 unique silver labels")
    for name, records in (("sources", sources), ("inputs", inputs)):
        require([r["dsld_id"] for r in records] == ids, f"ID/order mismatch: {name}")
    require([int(i) for i in draw["dsld_ids"]] == ids, "Draw ID/order mismatch")
    gold_ids = {r["dsld_id"] for r in rows(PILOT / "gold-1000.jsonl")}
    require(not gold_ids.intersection(ids), "Silver overlaps gold")

    traces = []
    raw_hash = hashlib.sha256()
    field_presence, top_presence, type_presence, type_blocks = (Counter() for _ in range(4))
    total_segments = total_paths = 0
    for row, source, prepared in zip(silver, sources, inputs):
        did = row["dsld_id"]
        lo = did // 1000 * 1000
        raw_path = ROOT / f"datasets/DSLD-full-database-JSON/{lo:06d}-{lo+999:06d}/{did}.json"
        raw_bytes = raw_path.read_bytes()
        raw_hash.update(raw_bytes)
        raw = json.loads(raw_bytes)
        require(raw["id"] == did, f"Raw ID mismatch: {did}")
        leaves = dict(strings(raw))
        normalized = {path: clean(value) for path, value in leaves.items()}
        types = {f"statements[{i}].notes": s.get("type") or ""
                 for i, s in enumerate(raw.get("statements", []))}
        top_presence.update(raw.keys())
        field_presence.update({family(path) for path in leaves})
        type_presence.update(set(types.values()))
        type_blocks.update(types.values())
        passage = source["passage"]
        segments = prepared["source_segments"]
        require(passage == prepared["passage"] == "\n".join(s["text"] for s in segments),
                f"Frozen passage/segment mismatch: {did}")
        spans, offset = [], 0
        for segment in segments:
            require(bool(segment["text"]) and bool(segment["paths"]), f"Empty segment: {did}")
            for path in segment["paths"]:
                require(normalized.get(path) == segment["text"], f"Raw segment mismatch: {did} {path}")
            spans.append((offset, offset + len(segment["text"]), segment))
            offset += len(segment["text"]) + 1
            total_segments += 1
            total_paths += len(segment["paths"])

        def location(path):
            return {"path": path, "field": family(path), "top_level_field": path.split("[", 1)[0].split(".", 1)[0],
                    "statement_type": types.get(path), "raw_text": leaves[path],
                    "normalized_text": normalized[path]}

        for kind in KINDS:
            for index, entry in enumerate(row[kind]):
                evidence = entry["evidence"]
                require(bool(evidence) and evidence in passage, f"Ungrounded evidence: {did} {kind}[{index}]")
                # Independent full-quotation search of every raw string field.
                matches = []
                for path, value in normalized.items():
                    if clean(evidence) in value:
                        matches.append({**location(path),
                                        "match_mode": "verbatim" if evidence in leaves[path] else "whitespace_normalized",
                                        "normalized_offsets": list(occurrences(value, clean(evidence)))})
                # Also trace quotes crossing a renderer-inserted newline. Keep
                # fragments distinct from complete-quotation matches above.
                projected = []
                for start in occurrences(passage, evidence):
                    end = start + len(evidence)
                    parts = []
                    for left, right, segment in spans:
                        a, b = max(start, left), min(end, right)
                        if a < b:
                            parts.append({"evidence_start": a - start, "evidence_end": b - start,
                                          "fragment": passage[a:b], "segment_start": a - left,
                                          "segment_end": b - left,
                                          "locations": [location(p) for p in segment["paths"]]})
                    require(parts and "\n".join(p["fragment"] for p in parts) == evidence,
                            f"Incomplete source coverage: {did} {kind}[{index}]")
                    projected.append({"passage_start": start, "passage_end": end, "parts": parts})
                contributing = {m["path"]: m for m in matches}
                for occurrence in projected:
                    for part in occurrence["parts"]:
                        for loc in part["locations"]:
                            contributing[loc["path"]] = loc
                traces.append({"dsld_id": did, "name": row["name"], "kind": kind,
                               "index": index, "text": entry["text"], "evidence": evidence,
                               "raw_file": str(raw_path.relative_to(ROOT)),
                               "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                               "status": "single_field" if matches else "cross_segment",
                               "fields": sorted({m["field"] for m in contributing.values()}),
                               "top_level_fields": sorted({m["top_level_field"] for m in contributing.values()}),
                               "statement_types": sorted({m["statement_type"] for m in contributing.values()
                                                          if m["statement_type"] is not None}),
                               "raw_matches": matches, "passage_occurrences": projected})
    require(raw_hash.hexdigest() == draw["input_sha256"]["raw_labels_concatenated_in_draw_order"],
            "Raw archive changed since silver draw")

    def stats(key, category):
        found = [t for t in traces if key in t[category]]
        result = {}
        for kind in KINDS:
            selected = [t for t in found if t["kind"] == kind]
            result.update({f"{kind}_claims": len(selected),
                           f"{kind}_labels": len({t["dsld_id"] for t in selected}),
                           f"{kind}_exclusive_claims": sum(t[category] == [key] for t in selected)})
        result["examples"] = [{k: t[k] for k in ("dsld_id", "kind", "index", "evidence")}
                              for t in found[:3]]
        return result

    report = {
        "method": "Case-sensitive evidence matching across every raw string leaf after whitespace collapse; "
                  "frozen source segments independently verified against raw fields. Cross-segment quotations "
                  "are mapped as ordered fragments, not reported as complete single-field matches. Counts "
                  "include each claim once per contributing field/type, including cross-segment fragments; "
                  "columns are not additive. Exclusive means no other field/type contributes to the quote. "
                  "All indices and character offsets are zero-based; end offsets are exclusive.",
        "labels": len(ids), "overlap_with_gold": 0,
        "totals": {f"{k}_claims": sum(t["kind"] == k for t in traces) for k in KINDS},
        "labels_with_claims": {k: sum(bool(r[k]) for r in silver) for k in KINDS},
        "validation": {"frozen_hashes_verified": True, "raw_draw_hash_verified": True,
                       "source_segments_verified": total_segments, "source_paths_verified": total_paths,
                       "claims_traced": len(traces), "unmatched_claims": 0,
                       "complete_quote_in_single_raw_field": sum(t["status"] == "single_field" for t in traces),
                       "claims_requiring_whitespace_normalization": sum(
                           bool(t["raw_matches"]) and all(m["match_mode"] != "verbatim" for m in t["raw_matches"])
                           for t in traces),
                       "cross_segment_claims": sum(t["status"] == "cross_segment" for t in traces)},
        "statement_types": {key: {"raw_labels": type_presence[key], "raw_statements": type_blocks[key],
                                   **stats(key, "statement_types")} for key in sorted(type_presence)},
        "raw_fields": {key: {"present_labels": top_presence[key], **stats(key, "top_level_fields")}
                       for key in sorted(top_presence)},
        "raw_text_fields": {key: {"present_labels": field_presence[key], **stats(key, "fields")}
                            for key in sorted(field_presence)},
        "cross_segment_examples": [t for t in traces if t["status"] == "cross_segment"],
        "input_sha256": {str(p.relative_to(ROOT)): digest(p)
                         for p in (silver_path, sources_path, inputs_path, manifest_path, draw_path)},
        "raw_labels_sha256_in_silver_order": raw_hash.hexdigest(),
        "limits": "Traces the recorded evidence, not semantic completeness or extraction accuracy. "
                  "Zero evidence matches do not establish that a field or statement type is irrelevant. "
                  "No model calls, no source edits, no clinical assessment.",
    }
    # All validation succeeds before any derived file is written.
    trace_path = PILOT / "silver-1000.field-trace.jsonl"
    trace_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces))
    csv_path = PILOT / "silver-1000.field-trace.csv"
    columns = ["dsld_id", "name", "kind", "index", "text", "evidence", "raw_file",
               "path", "field", "statement_type", "match_scope", "matched_text", "raw_text"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for trace in traces:
            base = {k: trace[k] for k in columns[:7]}
            full_paths = {m["path"] for m in trace["raw_matches"]}
            for match in trace["raw_matches"]:
                writer.writerow({**base, **{k: match[k] for k in columns[7:10]},
                                 "match_scope": "complete_quote", "matched_text": trace["evidence"],
                                 "raw_text": match["raw_text"]})
            seen = set()
            for occurrence in trace["passage_occurrences"]:
                for part in occurrence["parts"]:
                    for loc in part["locations"]:
                        key = (loc["path"], part["fragment"])
                        if loc["path"] not in full_paths and key not in seen:
                            writer.writerow({**base, **{k: loc[k] for k in columns[7:10]},
                                             "match_scope": "quote_fragment", "matched_text": part["fragment"],
                                             "raw_text": loc["raw_text"]})
                            seen.add(key)
    report["output_sha256"] = {p.name: digest(p) for p in (trace_path, csv_path)}
    (PILOT / "silver-1000.field-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps({k: result[k] for k in ("labels", "totals", "validation")}, indent=2))
