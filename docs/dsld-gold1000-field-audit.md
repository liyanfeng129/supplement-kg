# DSLD field audit against gold-1000

Checked all 1,000 gold labels against their frozen passages and original local DSLD JSON records. The reference contains 1,979 purpose claims and 2,302 risk claims. All 4,281 complete evidence quotations occur in `statements[].notes`; none occurs in another raw top-level field. This is a statement about the recorded quotations, not proof that other fields contain no relevant language.

**Recommendation:** retain free-text statements across all types, retain free-text ingredient notes/descriptions (including nested rows), and remove operational metadata from the extraction prompt. Filter verified boilerplate by its actual content, not by statement type. The existing three-type exclusion is not safe on this larger sample. Keep the raw records and IDs for provenance.

## Implemented extraction input (renderer version 2)

`label_text()` now omits product/brand/type/claim/target-group/ingredient-name
headers and statement-type prefixes. It retains statement bodies across **all**
types, plus ingredient notes and descriptions at every nesting depth. Exact
duplicate text is emitted once; `label_segments()` preserves every raw source
path. IDs and product names remain in output metadata. Structured operational
metadata, quantities and contact details were already outside the prompt.

The former three-type exclusion is removed. There is no keyword allowlist or
gold-dependent filtering at runtime: an absent gold quotation cannot establish
that a block has no purpose/risk language. Other raw free text, such as
`otheringredients.text` and contact text, remains outside this projection; this
audit does not certify those fields as irrelevant. Raw labels and frozen gold
files are unchanged.

Rebuild the 1,000 inputs and verify every gold quotation against its raw source:

```sh
.venv/bin/python datasets/pilot/prepare_gold1000_inputs.py > datasets/pilot/gold-1000.input-pruning-audit.json
```

The script writes `data/dsld/runs/gold1000-text-v2/raw/inputs.jsonl`. Each record
contains `dsld_id`, `name`, `passage`, `source_segments` with raw field paths,
and `gold_evidence_paths` linking each purpose/risk entry index to those paths.
Only `passage` is model input; the gold trace is audit metadata. The script
fails before writing if any gold evidence is lost. Use the pipeline's
`--records data/dsld/runs/gold1000-text-v2/raw/inputs.jsonl` option to select it;
existing frozen input files are not automatically rerendered.

The resulting **1,455,710 characters** retain all **4,281** gold quotations,
down **23.0%** from 1,890,693 characters. This reduction is smaller than the
39.2% frozen-passage ablation below because version 2 also restores omitted
statement types and adds ingredient free text. The report records source/output
hashes. No extraction model was rerun, so prediction quality remains unmeasured.

## Field removal versus field-name removal

Apply the selection to the structured label **before** joining retained text
into a passage. `label_segments()` performs that selection; `label_text()`
then joins only segment values with newlines. The audit's source paths and gold
references stay outside the passage sent to the LLM.

| Raw field | Action for the extraction passage |
|---|---|
| `statements[].notes` | Keep the value; omit the key `notes` and JSON wrappers. |
| Ingredient `notes` / `description`, including nested rows | Keep the values; omit field names and structural wrappers. |
| `statements[].type` | Drop the entire field, including its category value. A prefix such as `[Precautions re: All Other]` is the type **value**, not just a field name. |
| Product/brand/type/claim/target-group/ingredient-name headers | Version 2 drops names **and values**. Names-only removal is a separate comparison that retains the context values. |
| IDs, URLs, dates, source bookkeeping | Keep outside the model passage for joining and provenance. |

Removing names alone can preserve useful context at lower token cost. Removing
whole metadata fields saves more but can affect how a model interprets the
remaining text. Neither requires deleting the source record. There is no
per-label rule that drops a statement merely because no gold quotation matched
it: the reference does not exhaustively annotate every raw field.

Measure both choices with the actual configured local tokenizer:

```sh
.venv/bin/python datasets/pilot/compare_gold1000_inputs.py > datasets/pilot/gold-1000.token-pruning-audit.json
```

This comparison holds the frozen statement bodies constant while removing
header names, whole metadata fields, and statement-type values separately. A
final variant measures the current renderer, including its restored source
text. It reports both passage tokens and complete prompt tokens (system text
and chat formatting included), plus lost gold quotations for each variant.
The default uses the pipeline's JSON extraction prompt; `--format lines`
measures the other prompt format. Tokenization loads only the vocabulary and
does not run extraction.

Measured across 1,000 labels with the local GGUF tokenizer and the current
JSON system prompt (whose hash is stored in the report):

| Variant | Label tokens | Label-token reduction | Complete-prompt reduction |
|---|---:|---:|---:|
| Original frozen passages | 479,480 | — | — |
| Remove header names only; retain values | 455,806 | 4.9% | 1.3% |
| Drop entire metadata headers; retain statement types | 387,474 | 19.2% | 5.2% |
| Drop statement-type fields; retain metadata headers | 366,818 | 23.5% | 6.4% |
| Drop metadata headers and statement-type fields | 274,812 | 42.7% | 11.6% |
| Current renderer, including restored source text | 363,331 | 24.2% | 6.6% |

Every variant retains all 1,979 purpose and 2,302 risk quotations. The 42.7%
variant applies only to the historically filtered passages; it omits the raw
exceptions described below and should not replace the current renderer on
token savings alone. Fixed system/chat overhead makes complete-prompt savings
smaller than label-only savings. These are input token measurements, not output
token counts, inference latency measurements or extraction quality scores.

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

The reference is model-annotated, has prompt/cohort differences, and is drawn from the DSLD-only pilot rather than the entire archive. This analysis measures recorded evidence retention and supplies selected raw-content counterexamples; it is not a complete human adjudication of all free text. The original audit did not change production extraction; the subsequent renderer version 2 implementation is described above. Prompts and gold files remain unchanged.
