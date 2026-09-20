# DSLD field audit against gold-1000

Checked all 1,000 gold labels against their frozen passages and original local DSLD JSON records. The reference contains 1,979 purpose claims and 2,302 risk claims. All 4,281 complete evidence quotations occur in `statements[].notes`; none occurs in another raw top-level field. This is a statement about the recorded quotations, not proof that other fields contain no relevant language.

**Recommendation:** retain free-text statements across all types, retain free-text ingredient notes/descriptions (including nested rows), and remove operational metadata from the extraction prompt. Filter verified boilerplate by its actual content, not by statement type. The existing three-type exclusion is not safe on this larger sample. Keep the raw records and IDs for provenance.

## Where the gold evidence occurs

Counts below are claim entries whose complete evidence appears in a raw statement of that type, after the same whitespace collapse used by the passage renderer. A claim can match multiple types, so columns must not be summed. Raw counts include duplicate statements that were deduplicated before annotation. Exact quotation matching can miss equivalent wording; it does not measure semantic completeness.

| Raw statement type | Labels containing type | Purpose claims | Risk claims |
|---|---:|---:|---:|
| Brand IP Statement(s) re: (c), (TM), (SM) | 487 | 100 | 3 |
| FDA Disclaimer Statement | 789 | 0 | 0 |
| FDA Statement of Identity | 943 | 0 | 0 |
| Formula re: Contains | 642 | 381 | 76 |
| Formula re: Kosher | 78 | 0 | 0 |
| Formula re: Type | 84 | 35 | 4 |
| Formulation re: Does NOT Contain | 742 | 3 | 1 |
| Formulation re: Organic | 57 | 0 | 0 |
| Formulation re: Other | 344 | 507 | 2 |
| Formulation re: Vegetarian/Vegan | 196 | 2 | 0 |
| General Statements: All Other Content | 858 | 949 | 17 |
| General: Product/Version Code | 461 | 0 | 0 |
| Precautions re: All Other | 728 | 1 | 846 |
| Precautions re: Allergies | 280 | 0 | 271 |
| Precautions re: Children | 819 | 0 | 282 |
| Precautions re: Pregnant or Nursing or Prescription Medications | 700 | 1 | 888 |
| Seals/Symbols | 565 | 0 | 1 |
| Storage | 759 | 0 | 2 |
| Suggested/Recommended/Usage/Directions | 981 | 19 | 87 |

Purpose is concentrated in general statements and formulation text. Risk is concentrated in precautions, but also occurs in formulation, directions, brand/IP and even seals/symbols. For example, label **260680** places its medication/procedure consultation warning under `Seals/Symbols`. `Formulation re: Does NOT Contain` and `Formulation re: Vegetarian/Vegan` also contain gold purpose evidence. None of these is safe to discard as a whole type.

## What can be removed from the extraction input

Here “drop” means omit from the purpose/risk model input, not delete from the source archive or other KG tasks.

| Fields/content | Recommendation | Reason |
|---|---|---|
| `src`, `id`, `nhanesId`, `bundleName`, `upcSku`, `pdf`, `thumbnail`, `labelRelationships`, `events`, `entryDate`, `offMarket`, `hasOuterCarton` | Omit from prompt; keep identifiers/provenance outside it | Operational metadata; none supplies a gold evidence quotation. |
| Contact addresses, phone/email/web details; package counts; structured ingredient IDs, quantities, units and daily values | Omit from this explicit-claim extraction prompt | They are not the textual evidence being extracted. Still useful for other KG tasks. Do not extend this rule to arbitrary free-text notes. |
| `claims`, `productType`, `physicalState`, `targetGroups`, `userGroups`, brand and ingredient-name headers | Candidates for prompt removal | No gold evidence depends on these headers. Categories are context, not a substitute for explicit benefit/warning language. |
| `fullName` / product name | Optional short context; keep in output metadata | Gold evidence does not require it, but names can convey intended use and help resolve references. Removing it has not been tested with model predictions. |
| `statements[].type` prefixes | Candidate for prompt removal; retain source mapping | All gold quotations survive without prefixes. Type hints may still affect model performance. |
| `statements[].notes` | Keep across types, subject to content-based boilerplate removal | Relevant language appears under misleading types, including current skip types. |
| `ingredientRows[].notes`, `.description`, and recursive `nestedRows` free text | Retain/review | A nested note explicitly contains benefit language absent from the frozen passage; see label 29228 below. |
| Other unannotated free text, including `otheringredients.text` and contact free text | Not certified safe by this gold set | The gold annotators did not see these raw fields. Zero matching quotations is insufficient evidence for a blanket drop rule. |
| Exact duplicate statement text | Deduplicate; preserve all source locations | Keeps one complete copy of the evidence. Already done in the frozen passages. |
| Plain standalone disclaimer, generic “Dietary Supplement”, actual lot/version code, certification-only text, storage-only instructions | Drop only after checking the content | Do not remove an entire mixed block that also contains a benefit, warning or child-safety fallback. |

## Exceptions hidden by the original filtering

The gold was annotated from already-filtered passages. It therefore cannot certify the safety of the existing `SKIP_TYPES`. Reading the raw records reveals concrete counterexamples:

| Label | Raw location | Text / implication |
|---|---|---|
| 273770 | `General: Product/Version Code` | “Consult with a physician, pharmacist, naturopath or other qualified healthcare professional prior to taking dietary supplements.” This warning was excluded from the gold passage and is absent from its gold risk list. |
| 74782 | `FDA Disclaimer Statement` | Contains “Please consult your physician or the appropriately licensed professional before engaging in a program of exercise or nutritional supplementation.” The complete mixed disclaimer was excluded. Another physician-consultation warning survives elsewhere, but this type is demonstrably not pure boilerplate. |
| 7443 | `FDA Statement of Identity` | “Antioxidant Supplement”. More than a generic identity; this is purpose-like wording that needs the same annotation policy as other short benefit claims. |
| 29228 | `ingredientRows[5].nestedRows[0].notes` | “Nopal Cactus Note: (Organic)(TruGanic(TM))(supports endurance & blood sugar functions)”. Not in the frozen passage. Related product-level benefits are annotated elsewhere, so this demonstrates a source field containing purpose language, not necessarily a new distinct product-level claim. |
| 229138 | `Formulation re: Organic` | “What am I? I'm an organic pre-workout and energy powder.” Already in the frozen passage but not selected as a gold quotation; other energy benefits were selected. Zero evidence matches does not mean zero purpose language. |
| 175727 | `Formulation re: Organic` | A mixed organic statement includes “antioxidant activity-Cloves”; other antioxidant statements supply the gold evidence. |

`Formula re: Kosher` has no gold quotations; reviewed raw statements are mainly certification text (some also say dairy/contains gelatin). This supports removing certification-only text for this task, not treating the type as a universal safe filter. `Storage` matches two gold risk quotations containing child-safety language (180371, 180434); those exact quotations also survive in precautions. Removing storage loses no recorded quotation here, but keeping out of reach of children is explicitly allowed as a fallback by the frozen prompt.

## Measured input reductions

Reductions are aggregate **characters**, not measured tokens or inference costs, relative to the existing frozen passages. No extraction model was rerun. Rows are independent alternatives, not cumulative steps.

| Variant | Character reduction | Purpose quotations lost | Risk quotations lost |
|---|---:|---:|---:|
| Remove metadata headers; keep type prefixes | 16.3% | 0 | 0 |
| Remove type prefixes; keep metadata headers | 22.9% | 0 | 0 |
| Keep all statement bodies; remove both headers and prefixes | 39.2% | 0 | 0 |
| Statement bodies without Storage | 41.2% | 0 | 0 |
| Only General Statements + Precautions bodies | 70.8% | 1034 | 124 |
| Only Precautions bodies | 85.5% | 1977 | 141 |

The 39.2% reduction is a promising ablation for the existing benchmark. It does **not** fix upstream omissions. A future full-label extraction input should restore substantive text currently lost through type filtering and include ingredient free text. Evaluate that input separately and review the affected gold rows; do not silently rewrite the frozen reference.

## Reproduction and limits

```sh
python3 datasets/pilot/validate_gold1000.py
python3 datasets/pilot/audit_gold1000_fields.py > datasets/pilot/gold-1000.field-audit.json
```

The audit checks every frozen statement against the corresponding raw label, and every gold quotation against both. The JSON includes field/type counts, exclusive evidence counts, example quotations, input hashes, and verified raw exceptions. Both checks passed. The audit requires the local DSLD dump and makes no network/model calls.

The reference is model-annotated, has prompt/cohort differences, and is drawn from the DSLD-only pilot rather than the entire archive. This analysis measures recorded evidence retention and supplies selected raw-content counterexamples; it is not a complete human adjudication of all free text. Production extraction, prompts and gold files were not changed.
