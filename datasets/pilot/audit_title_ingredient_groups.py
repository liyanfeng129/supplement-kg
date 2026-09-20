"""Offline title-to-group candidate audit, not adjudicated main ingredients.

Run from any directory with Python 3. Writes a CSV and evidence JSON alongside
this script. Uses only sampled IDs and raw DSLD records, never gold claims.
"""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "datasets/pilot"
# Exclude generic totals as targets, but still traverse their children.
# Named sources such as Whey Protein, Fish Oil and Pectin remain eligible.
GENERIC_GROUPS = {
    "calories", "carbohydrate", "sugar", "fat (unspecified)",
    "saturated fat", "trans fat", "monounsaturated fat", "polyunsaturated fat",
    "fiber (unspecified)", "protein (unspecified)", "protein", "cholesterol",
}


def normalize(text):
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def walk(rows, prefix="ingredientRows"):
    for index, row in enumerate(rows):
        path = f"{prefix}[{index}]"
        yield path, row
        yield from walk(row.get("nestedRows") or [], path + ".nestedRows")


def match_label(raw, placeholders):
    title = raw.get("fullName") or ""
    normalized_title = " " + normalize(title) + " "
    matches, excluded, meaningful = [], [], set()
    for path, row in walk(raw.get("ingredientRows") or []):
        group = (row.get("ingredientGroup") or "").strip()
        if not group or group.casefold() in placeholders:
            continue
        generic = group.casefold() in GENERIC_GROUPS
        if not generic:
            meaningful.add(group.casefold())
        terms = [("ingredientGroup", row["ingredientGroup"]),
                 ("name", row.get("name") or "")]
        terms += [(f"alternateNames[{i}]", value)
                  for i, value in enumerate(row.get("alternateNames") or [])
                  if isinstance(value, str)]
        for field, term in terms:
            normalized = normalize(term)
            # Short chemical symbols are too ambiguous (e.g. Mg versus mg).
            if len(normalized) < 3 or " " + normalized + " " not in normalized_title:
                continue
            evidence = dict(group=group, row_path=path,
                            ingredient_id=row.get("ingredientId"),
                            ingredient_name=row.get("name"),
                            matched_field=field, matched_value=term)
            (excluded if generic else matches).append(evidence)

    def groups_for(evidence):
        # Whitespace/case differences do not create extra group identities.
        return sorted({m["group"].casefold(): m["group"] for m in evidence}.values())

    groups = groups_for(matches)
    direct_groups = groups_for([m for m in matches if m["matched_field"] == "ingredientGroup"])
    flags = []
    if re.search(r"\bflavou?r(?:ed|ing)?\b", title, re.I):
        flags.append("title_mentions_flavor")
    if re.search(r"\b(?:free|without|no)\b", title, re.I):
        flags.append("title_has_possible_exclusion")
    if re.search(r"\b(?:with|plus|and)\b|[&+]", title, re.I):
        flags.append("title_has_possible_combination")
    if len(meaningful) > len(groups):
        flags.append("additional_groups_not_matched_in_title")
    if any(re.search(r"\bflavou?r(?:ed|ing)?\b", m["ingredient_name"] or "", re.I)
           for m in matches):
        flags.append("matched_row_mentions_flavor")
    return dict(dsld_id=raw["id"], product=title,
                status=("single_group_candidate" if len(groups) == 1 else
                        "multiple_group_candidates" if groups else "no_title_match"),
                candidate_main_group=groups[0] if len(groups) == 1 else None,
                candidate_groups=groups, direct_group_matches=direct_groups,
                meaningful_group_count=len(meaningful), review_flags=flags,
                matches=matches, excluded_generic_matches=excluded)


def main():
    source = PILOT / "gold-1000.sources.jsonl"
    group_audit = PILOT / "gold-1000.ingredient-group-audit.json"
    placeholders = set(json.loads(group_audit.read_text())["placeholder_values"])
    ids = [json.loads(line)["dsld_id"] for line in source.read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 1000
    results, raw_hashes = [], {}
    for did in ids:
        lo = did // 1000 * 1000
        path = ROOT / f"datasets/DSLD-full-database-JSON/{lo:06d}-{lo+999:06d}/{did}.json"
        content = path.read_bytes()
        raw = json.loads(content)
        assert raw["id"] == did
        results.append(match_label(raw, placeholders))
        raw_hashes[str(did)] = hashlib.sha256(content).hexdigest()
    counts = dict(Counter(row["status"] for row in results))
    counts["labels"] = len(results)
    counts["any_direct_group_match"] = sum(bool(r["direct_group_matches"]) for r in results)
    counts["single_direct_group_match"] = sum(len(r["direct_group_matches"]) == 1 for r in results)
    counts["any_match_including_names_and_aliases"] = sum(bool(r["candidate_groups"]) for r in results)
    counts["single_candidate_with_matched_flavor_row"] = sum(
        r["status"] == "single_group_candidate" and "matched_row_mentions_flavor" in r["review_flags"]
        for r in results)
    fields = ["dsld_id", "product", "status", "candidate_main_group", "candidate_groups",
              "direct_group_matches", "meaningful_group_count", "review_flags"]
    with (PILOT / "gold-1000.title-ingredient-candidates.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False)
                             if isinstance(row[k], list) else row[k] for k in fields})
    report = dict(
        method="Casefolded whole-token phrase matching after punctuation normalization; minimum 3 characters. Group, row name and raw row aliases only. No inferred aliases, amount ranking or semantic adjudication. Every result is a candidate, including unique matches; do not automatically attribute product claims to it.",
        counts=counts, generic_groups=sorted(GENERIC_GROUPS),
        placeholder_groups=sorted(placeholders),
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (source, group_audit)},
        raw_record_sha256=raw_hashes, labels=results)
    (PILOT / "gold-1000.title-ingredient-candidates.audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
