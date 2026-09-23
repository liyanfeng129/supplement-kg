# DSLD field audit against silver-1000

All **1,000 silver labels** were traced to the original local DSLD JSON dump.
All **1,456 purpose** and **1,863 risk** entries (3,319 total) are grounded in
`statements[].notes`. No extracted quotation matched any other raw text field,
including ingredient notes/descriptions, product names, or structured claim
categories. This describes the extracted evidence, not every possible claim
in the raw labels.

Of these entries, **3,318 have their complete quotation in an individual raw
field** after whitespace normalization. The remaining purpose quotation spans
two statement fields, detailed below. There are **zero unmatched entries**.
There are 637 labels with purpose entries and 814 with risk entries.

## Where the evidence occurs

`statements[].notes` contains the actual text. `statements[].type` supplies the
category shown below; it is provenance metadata, not the quotation itself.
Counts are distinct claim entries per contributing type, retaining alternative
locations where the raw label repeats text. A claim can count in multiple rows,
so the columns must not be summed. The single cross-statement quotation counts
under both contributing types, even though neither contains the whole quote.

| Raw statement type | Labels containing type | Purpose entries | Risk entries |
|---|---:|---:|---:|
| Brand IP Statement(s) re: (c), (TM), (SM) | 457 | 81 | 1 |
| FDA Disclaimer Statement | 762 | 0 | 0 |
| FDA Statement of Identity | 936 | 3 | 0 |
| Formula re: Contains | 647 | 229 | 17 |
| Formula re: Kosher | 69 | 0 | 0 |
| Formula re: Type | 77 | 21 | 0 |
| Formulation re: Does NOT Contain | 750 | 4 | 0 |
| Formulation re: Organic | 51 | 0 | 0 |
| Formulation re: Other | 319 | 323 | 0 |
| Formulation re: Vegetarian/Vegan | 225 | 3 | 0 |
| General Statements: All Other Content | 849 | 787 | 8 |
| General: Product/Version Code | 477 | 0 | 0 |
| Precautions re: All Other | 715 | 0 | 729 |
| Precautions re: Allergies | 294 | 0 | 86 |
| Precautions re: Children | 786 | 0 | 271 |
| Precautions re: Pregnant or Nursing or Prescription Medications | 697 | 0 | 855 |
| Seals/Symbols | 556 | 3 | 0 |
| Storage | 757 | 0 | 1 |
| Suggested/Recommended/Usage/Directions | 986 | 8 | 23 |

Purpose evidence is concentrated in general statements and formulation text;
risk evidence is concentrated in precautions. Both also appear in other
categories, so category names alone are insufficient for selecting source text.

## Exact locations worth noting

All array indices below are zero-based, matching the original JSON.

| DSLD ID | Entry | Raw path | Type and evidence |
|---|---|---|---|
| 12707 | `purpose[0]` | `statements[6].notes` | **FDA Statement of Identity:** “MONSTER MILK(TM) is designed to help you build bigger, stronger muscles while staying lean” |
| 23627 | `purpose[1]` | `statements[11].notes` | **FDA Statement of Identity:** “A Dietary Supplement to Support Thyroid Function and Health” |
| 30461 | `purpose[0]` | `statements[19].notes` | **Seals/Symbols:** “100% Pure All Natural Formula SUPPORTS WEIGHT LOSS FOR ALL BODY TYPES” |
| 42446 | `purpose[2]` | `statements[10].notes` | **Seals/Symbols:** “SEASONAL RESPIRATORY SUPPORT” |
| 54990 | `purpose[0]` | `statements[1].notes`, `statements[2].notes` | **Does NOT Contain** and **Vegetarian/Vegan:** “Natrol Melatonin is a 100% vegetarian, drug-free sleep aid for relief of occasional sleeplessness.” |
| 25803 | `risk[0]` | `statements[5].notes`, `statements[6].notes` | **Storage** and **Precautions re: Children:** “Caution: Keep out of direct light and out of the reach of children.” |

Silver therefore directly confirms that `FDA Statement of Identity` can supply
extracted purpose evidence. The first two quotations above occur only under
that type. A third identity quotation, on label 58454, also appears under
`Formula re: Type`. Dropping the formerly excluded identity/disclaimer/product
code types would lose two recorded purpose quotations in silver.

Label **43806**, `purpose[0]`, is the cross-statement case. Its evidence joins:

1. `statements[11].notes`, **Formula re: Contains**:
   “~ Concentrated Fish Oil Omega-3 fatty acids”
2. `statements[12].notes`, **General Statements: All Other Content**:
   “~ Exclusive process to preserve whole oil Omega-3 concentrations ~ Omega-3 fatty acids help support healthy heart, joint, immune, skin, brain & memory and nervous systems”

The frozen renderer inserts a newline between those fields. The complete
quotation exists verbatim in the frozen passage, but not in either raw field
alone. The benefit wording occurs in the second field; the first supplies
context included in the annotator's evidence. The trace preserves the original
quotation and maps its fragments separately.

## Saved results and reproduction

- [Summary JSON](../datasets/pilot/silver-1000.field-audit.json): counts by raw
  field and statement type, exclusive counts, examples, hashes, and validation.
- [Per-claim JSONL](../datasets/pilot/silver-1000.field-trace.jsonl): 3,319 records,
  each containing DSLD ID, claim kind/index, claim text, unchanged evidence,
  raw file/hash, all matching paths and types, raw text, and passage offsets.
- [Location CSV](../datasets/pilot/silver-1000.field-trace.csv): one row per
  claim/location; `match_scope` distinguishes complete quotations from the
  cross-statement fragments. Multiple rows can refer to one claim.

Reproduce using only Python's standard library and the local dump:

```sh
python3 datasets/pilot/audit_silver1000_fields.py
```

The audit independently searches **every raw string field**, rather than only
the fields selected by the renderer. It also verifies all **18,248 frozen
source segments** and **18,623 source paths** against raw text, checks ID/order
consistency, verifies frozen input and raw-dump hashes against the manifests,
and confirms zero overlap with gold. All checks pass before outputs are written.

Matching is case-sensitive. Raw whitespace is collapsed as in the renderer;
268 complete quotations require this normalization to match any raw location.
All evidence remains verbatim in its frozen passage. All indices and offsets
are zero-based; end offsets are exclusive. Repeated raw locations are retained:
135 entries have a complete-quotation match in more than one raw field location.

This audit makes no model calls and does not alter silver, gold, their frozen
passages, or the extraction pipeline. It verifies provenance, not extraction
accuracy or completeness. No ingredient-note quotations were extracted here,
but that does not establish that ingredient free text is irrelevant. Silver's
annotator and renderer differ from gold's, so frequency differences between
the sets cannot be attributed to renderer changes alone.
