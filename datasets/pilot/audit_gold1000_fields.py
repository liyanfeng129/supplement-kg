"""Map frozen gold evidence to DSLD fields; no model calls or input changes.

Run from any directory with Python 3. Outputs JSON to stdout. Requires the
local DSLD dump. Counts measure evidence retention, not model accuracy or
completeness of the annotations. Raw whitespace is collapsed as in label_text.
"""

import json
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "datasets/pilot"
SKIPPED = {"FDA Statement of Identity", "FDA Disclaimer Statement",
           "General: Product/Version Code"}


def strings(value):
    if isinstance(value, str):
        yield " ".join(value.split())
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit():
    gold = rows(PILOT / "gold-1000.jsonl")
    sources = {r["dsld_id"]: r for r in rows(PILOT / "gold-1000.sources.jsonl")}
    assert len(gold) == len(sources) == len({r['dsld_id'] for r in gold}) == 1000
    records, all_types, all_fields = [], set(), set()
    for row in gold:
        did = row["dsld_id"]
        lo = did // 1000 * 1000
        path = ROOT / f"datasets/DSLD-full-database-JSON/{lo:06d}-{lo+999:06d}/{did}.json"
        raw = json.loads(path.read_text())
        assert raw["id"] == did
        passage = sources[did]["passage"]
        header, marker, body = passage.partition("\nLABEL STATEMENTS:")
        assert marker
        body = body.lstrip("\n")
        statements = []
        for line in body.splitlines():
            match = re.fullmatch(r"\[([^\]]*)\] (.*)", line)
            assert match, (did, line)
            statements.append((match[1], match[2]))
        raw_statements = [(s.get("type") or "", " ".join((s.get("notes") or "").split()))
                          for s in raw.get("statements", [])]
        assert all(s in raw_statements for s in statements), did
        all_types.update(t for t, _ in raw_statements)
        all_fields.update(raw)
        claims = []
        for kind in ("purpose", "risk"):
            for index, entry in enumerate(row[kind]):
                evidence = entry["evidence"]
                assert evidence and evidence in passage
                frozen_types = {t for t, text in statements if evidence in text}
                raw_types = {t for t, text in raw_statements if evidence in text}
                assert frozen_types and raw_types, (did, evidence)
                fields = {k for k, value in raw.items()
                          if any(evidence in text for text in strings(value))}
                claims.append(dict(kind=kind, index=index, evidence=evidence,
                                   frozen_types=frozen_types, raw_types=raw_types,
                                   fields=fields))
        records.append(dict(id=did, raw=raw, header=header, passage=passage,
                            statements=statements, raw_statements=raw_statements,
                            claims=claims))

    def stats(key, mode):
        matches = [(r, c) for r in records for c in r["claims"] if key in c[mode]]
        exclusive = [(r, c) for r, c in matches if c[mode] == {key}]
        return {
            **{kind + "_claims": sum(c["kind"] == kind for _, c in matches)
               for kind in ("purpose", "risk")},
            **{kind + "_labels": len({r["id"] for r, c in matches if c["kind"] == kind})
               for kind in ("purpose", "risk")},
            "exclusive_claims": len(exclusive),
            "exclusive_labels": len({r["id"] for r, _ in exclusive}),
            "examples": [dict(dsld_id=r["id"], kind=c["kind"], evidence=c["evidence"])
                         for r, c in (exclusive or matches)[:3]],
        }

    type_stats = {}
    for stype in sorted(all_types):
        type_stats[stype] = dict(
            raw_labels=sum(any(t == stype for t, _ in r["raw_statements"]) for r in records),
            raw_statements=sum(t == stype for r in records for t, _ in r["raw_statements"]),
            frozen_labels=sum(any(t == stype for t, _ in r["statements"]) for r in records),
            excluded_before_annotation=stype in SKIPPED,
            **stats(stype, "frozen_types"),
            raw_evidence=stats(stype, "raw_types"),
        )
    field_stats = {key: dict(present_labels=sum(key in r["raw"] for r in records),
                             **stats(key, "fields")) for key in sorted(all_fields)}

    def variant(keep):
        missed = [(r, c) for r in records for c in r["claims"]
                  if not any(keep(t) and c["evidence"] in text for t, text in r["statements"])]
        chars = sum(len("\n".join(text for t, text in r["statements"] if keep(t))) for r in records)
        return dict(characters=chars,
                    reduction_vs_frozen_passages=round(1 - chars / total_chars, 6),
                    **{kind + "_claims_lost": sum(c["kind"] == kind for _, c in missed)
                       for kind in ("purpose", "risk")},
                    labels_losing_evidence=len({r["id"] for r, _ in missed}))

    total_chars = sum(len(r["passage"]) for r in records)
    evidence_types = {t for r in records for c in r["claims"] for t in c["frozen_types"]}
    variants = {
        "all_statement_bodies_no_headers_or_type_prefixes": variant(lambda t: True),
        "only_types_with_observed_gold_evidence_in_this_sample": variant(lambda t: t in evidence_types),
        "only_general_and_precautions": variant(lambda t: t.startswith(("General Statements:", "Precautions"))),
        "drop_storage": variant(lambda t: t != "Storage"),
        "only_precautions": variant(lambda t: t.startswith("Precautions")),
    }
    for name, render in (
        ("remove_metadata_headers_keep_type_prefixes",
         lambda r: "\n".join(f"[{t}] {text}" for t, text in r["statements"])),
        ("remove_type_prefixes_keep_metadata_headers",
         lambda r: r["header"] + "\nLABEL STATEMENTS:\n" +
                   "\n".join(text for _, text in r["statements"])),
    ):
        chars = sum(len(render(r)) for r in records)
        assert all(c["evidence"] in render(r) for r in records for c in r["claims"])
        variants[name] = dict(characters=chars,
                              reduction_vs_frozen_passages=round(1-chars/total_chars, 6),
                              purpose_claims_lost=0, risk_claims_lost=0,
                              labels_losing_evidence=0)

    # Concrete exceptions discovered by reviewing the raw fields. These are
    # illustrative counterexamples, not automatically adjudicated new gold.
    review_cases = []
    for did, path, phrase in (
        (74782, "statements[].notes [FDA Disclaimer Statement]", "Please consult your physician"),
        (273770, "statements[].notes [General: Product/Version Code]", "Consult with a physician"),
        (7443, "statements[].notes [FDA Statement of Identity]", "Antioxidant Supplement"),
        (229138, "statements[].notes [Formulation re: Organic]", "I'm an organic pre-workout and energy powder"),
        (175727, "statements[].notes [Formulation re: Organic]", "antioxidant activity-Cloves"),
        (29228, "ingredientRows[5].nestedRows[0].notes", "supports endurance & blood sugar functions"),
    ):
        record = next(r for r in records if r["id"] == did)
        if path.startswith("statements"):
            stype = path.split(" [", 1)[1][:-1]
            note = next(text for t, text in record["raw_statements"] if t == stype and phrase in text)
        else:
            note = " ".join(record["raw"]["ingredientRows"][5]["nestedRows"][0]["notes"].split())
        assert phrase in note
        review_cases.append(dict(dsld_id=did, path=path, text=note,
                                 text_in_frozen_passage=note in record["passage"]))
    return dict(
        method="Case-sensitive verbatim evidence matching; collapse raw statement whitespace as in renderer. Multi-type matches count in each type. Exclusive means no alternative type/field contains the complete evidence span. No inference about model accuracy.",
        labels=len(records),
        totals={kind + "_claims": sum(len(r[kind]) for r in gold) for kind in ("purpose", "risk")},
        frozen_passage_characters=total_chars,
        statement_types=type_stats, raw_fields=field_stats, variants=variants,
        raw_review_examples=review_cases,
        input_sha256={name: hashlib.sha256((PILOT / name).read_bytes()).hexdigest()
                      for name in ("gold-1000.jsonl", "gold-1000.sources.jsonl")},
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
