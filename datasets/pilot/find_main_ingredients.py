"""Cell-by-cell DSLD main-ingredient analysis; run with Python 3.

Uses the audited 1,000 IDs only as a sampling index, never their annotations.
`raw_labels` may instead be supplied by a notebook or runpy for another sample.
Outputs are held in `main_ingredients` and `ingredient_audit`; no input changes.
"""

# %% 1. Load raw labels for the audited sample (skip when raw_labels exists).
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

if "raw_labels" not in globals():
    repo = Path(__file__).resolve().parents[2]
    sample_ids = [json.loads(line)["dsld_id"] for line in
                  (repo / "datasets/pilot/gold-1000.sources.jsonl").read_text().splitlines()]
    raw_labels = []
    for did in sample_ids:
        lo = did // 1000 * 1000
        path = repo / f"datasets/DSLD-full-database-JSON/{lo:06d}-{lo+999:06d}/{did}.json"
        raw_labels.append(json.loads(path.read_text()))

# %% 2. Define compatible mass units and explicit aggregate-row exclusions.
MASS_TO_MG = {
    "g": 1000, "gram(s)": 1000, "grams": 1000, "gram": 1000,
    "mg": 1, "milligram(s)": 1, "milligrams": 1,
    "mcg": .001, "µg": .001, "μg": .001, "microgram(s)": .001,
    "ng": .000001,
}
# Exact names: do not exclude named ingredients such as whey protein isolate,
# cane sugar, fish oil, or a specific fiber merely because of a keyword.
AGGREGATES = {
    "calories", "total calories", "calories from fat", "fat calories",
    "calories from carbohydrates", "total carbohydrates", "total carbohydrate",
    "carbohydrates", "other carbohydrates", "total fat", "fat", "fats",
    "saturated fat", "sat. fat", "trans fat", "polyunsaturated fat",
    "monounsaturated fat", "unsaturated fat", "sugar", "sugars", "total sugars",
    "added sugars", "includes added sugar", "sugar alcohol", "sugar alcohols",
    "protein", "proteins", "complete protein", "dietary fiber", "fiber",
    "soluble fiber", "insoluble fiber",
}

# %% 3. Separate individual ingredients, blends, aggregate rows and ambiguities.
ingredient_audit = {}
for raw in raw_labels:
    audit = {"ingredients": [], "excluded": [], "unresolved": []}
    stack = [(row, f"ingredientRows[{i}]", False)
             for i, row in reversed(list(enumerate(raw.get("ingredientRows") or [])))]
    while stack:
        row, path, in_blend = stack.pop()
        name = " ".join((row.get("name") or "").split())
        children = row.get("nestedRows") or []
        is_blend = (row.get("category") == "blend"
                    or bool(re.search(r"\bblend\b", name, re.I)))
        if name.casefold() in AGGREGATES:
            audit["excluded"].append({"name": name, "path": path})
        elif is_blend and children:
            stack.extend((child, f"{path}.nestedRows[{i}]", True)
                         for i, child in reversed(list(enumerate(children))))
        elif is_blend or children or not name:
            reason = ("blend has no listed components" if is_blend else
                      "non-blend nesting needs interpretation" if children else
                      "missing ingredient name")
            audit["unresolved"].append({"name": name, "path": path, "reason": reason})
        else:
            audit["ingredients"].append({"name": name, "path": path,
                                          "in_blend": in_blend, "row": row})
    ingredient_audit[raw["id"]] = audit

# %% 4. Resolve sole eligible ingredients without reading any quantities.
work_counts = Counter()
for audit in ingredient_audit.values():
    audit.update({"comparisons": [], "rejected_quantities": [], "decision": None,
                  "processing_path": None})
    ingredients = audit["ingredients"]
    if (len(ingredients) == 1 and not audit["unresolved"]
            and not ingredients[0]["in_blend"]):
        audit["decision"] = (ingredients[0]["name"], None, "sole_eligible_ingredient")
        audit["processing_path"] = "sole_without_amount_scan"
    elif not ingredients:
        audit["decision"] = (None, None, "unknown_no_comparable_amounts")
        audit["processing_path"] = "no_eligible_ingredients"

# %% 5. Check amount availability only for unresolved labels; do not convert yet.
for raw in raw_labels:
    audit = ingredient_audit[raw["id"]]
    if audit["decision"] is not None:
        continue
    serving_sizes = {s.get("order"): s for s in raw.get("servingSizes") or []}
    prepared = []
    for ingredient in audit["ingredients"]:
        for q in ingredient["row"].get("quantity") or []:
            work_counts["quantity_records_scanned"] += 1
            amount = q.get("quantity")
            factor = MASS_TO_MG.get((q.get("unit") or "").strip().casefold())
            if not (q.get("operator") == "=" and factor is not None
                    and isinstance(amount, (int, float)) and not isinstance(amount, bool)
                    and amount >= 0):
                audit["rejected_quantities"].append({"name": ingredient["name"],
                                                     "path": ingredient["path"], "quantity": q})
                continue
            basis_quantity = q.get("servingSizeQuantity")
            basis_unit = (q.get("servingSizeUnit") or "").strip().casefold()
            serving = serving_sizes.get(q.get("servingSizeOrder"), {})
            # Fill a missing basis only from a fixed serving definition.
            if (basis_quantity is None and serving.get("minQuantity") is not None
                    and serving.get("minQuantity") == serving.get("maxQuantity")
                    and (not basis_unit or basis_unit ==
                         (serving.get("unit") or "").strip().casefold())):
                basis_quantity = serving["minQuantity"]
                basis_unit = (serving.get("unit") or "").strip().casefold()
            usable = (isinstance(basis_quantity, (int, float))
                      and basis_quantity > 0 and basis_unit not in {"", "np", "not present"}
                      and q.get("servingSizeOrder") is not None)
            if not usable:
                audit["rejected_quantities"].append({"name": ingredient["name"],
                                                     "path": ingredient["path"], "quantity": q})
                continue
            prepared.append({"name": ingredient["name"], "path": ingredient["path"],
                             "amount": amount, "factor": factor, "quantity": q,
                             "basis": (q["servingSizeOrder"], basis_quantity, basis_unit)})
    audit["prepared"] = prepared
    if not prepared:
        audit["decision"] = (None, None, "unknown_no_comparable_amounts")
        audit["processing_path"] = "no_usable_mass"

# %% 6. Resolve a single positive usable amount without conversion or ranking.
for audit in ingredient_audit.values():
    if audit["decision"] is not None:
        continue
    prepared = audit["prepared"]
    if len(prepared) == 1 and prepared[0]["amount"] > 0:
        candidate = prepared[0]
        if len(audit["ingredients"]) == 1 and not audit["unresolved"]:
            audit["decision"] = (candidate["name"], None, "largest_complete_comparable_mass")
        else:
            audit["decision"] = (None, candidate["name"], "largest_among_quantified_only")
        audit["processing_path"] = "single_usable_amount_without_ranking"
        audit["single_amount_evidence"] = candidate

# %% 7. Convert and compare only the remaining labels.
for audit in ingredient_audit.values():
    if audit["decision"] is not None:
        continue
    audit["processing_path"] = "serving_comparison"
    groups = defaultdict(lambda: defaultdict(set))
    for candidate in audit["prepared"]:
        order, basis_quantity, basis_unit = candidate["basis"]
        # Normalize mass-based serving units as well as ingredient units.
        if basis_unit in MASS_TO_MG:
            basis_quantity *= MASS_TO_MG[basis_unit]
            basis_unit = "mg"
        basis = (order, basis_quantity, basis_unit)
        groups[basis][candidate["path"]].add(round(candidate["amount"] * candidate["factor"], 12))
        work_counts["ingredient_mass_conversions"] += 1
    by_path = {i["path"]: i for i in audit["ingredients"]}
    for basis, quantities in groups.items():
        # Conflicting amounts for the same ingredient/basis require review.
        conflict = any(len(values) != 1 for values in quantities.values())
        amounts = {path: next(iter(values)) for path, values in quantities.items()
                   if len(values) == 1}
        if len(amounts) <= 1:
            maximum = next(iter(amounts.values()), 0)
            leaders = list(amounts)
        else:
            work_counts["groups_requiring_mass_ranking"] += 1
            maximum = max(amounts.values())
            leaders = [path for path, amount in amounts.items() if amount == maximum]
        complete = (not audit["unresolved"] and not conflict
                    and len(amounts) == len(by_path))
        audit["comparisons"].append({
            "basis": list(basis), "complete": complete, "conflict": conflict,
            "positive_maximum": maximum > 0,
            "leaders": [{"name": by_path[p]["name"], "path": p, "mg": maximum}
                        for p in leaders] if maximum > 0 else [],
            "quantified_count": len(amounts), "ingredient_count": len(by_path),
        })

# %% 8. Combine early decisions with results from the remaining comparisons.
main_ingredients = []
for raw in raw_labels:
    audit = ingredient_audit[raw["id"]]
    ingredients = audit["ingredients"]
    main_name = None
    quantified_name = None
    complete = [c for c in audit["comparisons"] if c["complete"]]
    partial = [c for c in audit["comparisons"] if not c["conflict"] and c["leaders"]]
    if audit["decision"] is not None:
        main_name, quantified_name, status = audit["decision"]
    elif complete and len(complete) == len(audit["comparisons"]):
        leader_paths = {x["path"] for c in complete for x in c["leaders"]}
        if all(len(c["leaders"]) == 1 for c in complete) and len(leader_paths) == 1:
            main_name = complete[0]["leaders"][0]["name"]
            status = "largest_complete_comparable_mass"
        else:
            status = "unknown_tie_or_serving_disagreement"
    elif partial:
        leader_paths = {x["path"] for c in partial for x in c["leaders"]}
        if all(len(c["leaders"]) == 1 for c in partial) and len(leader_paths) == 1:
            quantified_name = partial[0]["leaders"][0]["name"]
            status = "largest_among_quantified_only"
        else:
            status = "unknown_tie_or_serving_disagreement"
    else:
        status = "unknown_no_comparable_amounts"
    main_ingredients.append({
        "dsld_id": raw["id"], "product": raw.get("fullName"),
        "main_ingredient": main_name,
        "largest_among_quantified": quantified_name, "status": status,
        "eligible_ingredient_count": len(ingredients),
        "unresolved_structure_count": len(audit["unresolved"]),
        "excluded_aggregate_count": len(audit["excluded"]),
    })

# %% 9. Inspect the results and avoided work.
if __name__ == "__main__":
    print(json.dumps(dict(Counter(r["status"] for r in main_ingredients)), indent=2))
    print(json.dumps(dict(Counter(a["processing_path"] for a in ingredient_audit.values())), indent=2))
    print(json.dumps(dict(work_counts), indent=2))
