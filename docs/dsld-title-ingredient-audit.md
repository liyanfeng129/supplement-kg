# Product names as evidence for main ingredients

For purpose/risk attribution, define the main ingredient as the ingredient
that identifies the product in its name and label context. A mass winner is a
different attribute. Use product-name interpretation first and verify the
proposed identity against that label's ingredient rows and groups.

The existing ingredient-group audit covers 1,000 labels and 9,172 rows,
including nested rows. All labels have a populated group somewhere, but 251
have placeholder groups and eight have only missing/placeholder groups.
Presence of a group does not establish that it is the main ingredient.

## Measured title matching

`datasets/pilot/gold-1000.title-ingredient-candidates.csv` has one row for each
of the 1,000 sampled labels. The companion `.audit.json` preserves matching
row paths, DSLD ingredient IDs, original names, matched fields, excluded
generic matches, input hashes and review flags.

The baseline compares whole-token phrases after case and punctuation
normalization. It first matches group names, then also accepts ingredient row
names and aliases actually recorded on that row, mapping these back to the
row's group. It excludes missing/placeholder groups and an explicit list of
generic nutrition groups. It traverses nested rows and excludes
`otheringredients`. Terms shorter than three characters are skipped to avoid
chemical-symbol/dosage collisions such as `Mg` and `mg`.

| Result | Labels |
|---|---:|
| At least one direct ingredient-group name match | 390 |
| At least one match after adding row names and recorded aliases | 454 |
| Exactly one candidate group with the expanded matching | 423 |
| Multiple candidate groups | 31 |
| No eligible title match | 546 |

The last three rows partition the sample. These are candidate coverage
counts, **not accuracy or confirmed main-ingredient counts**. They do not
establish that most titles identify a main ingredient, nor disprove that
hypothesis: this baseline misses unrecorded abbreviations, chemical-form
variants, joined words and implicit names. One lexical match can also miss
another ingredient named in the same title. No gold purpose/risk annotations
were used to generate the candidates.

## Examples checked against raw labels

| DSLD ID and title | Interpretation |
|---|---|
| 709, Huperzine A | Row Huperzine A maps to group Huperzine. Keep the precise row identity alongside the broader group. |
| 885, Spirulina Blue-Green Algae 500 mg | Spirulina is a row name; Blue-Green Algae is its group. Both title matches point to the same row. |
| 10932, Pycnogenol 30 mg | The title matches row Pycnogenol, whose group is maritime Pine. Group-name matching alone misses it. |
| 20061, Clinical Strength Lutein & Zeaxanthin | Two named groups. Preserve both candidates; selecting one by mass would discard the title's combination. |
| 10975, Maximum Strength Glucosamine MSM Complex | Literal matching finds only MSM/Methylsulfonylmethane, while the rows also contain Glucosamine Sulfate. One match does not prove one main ingredient. |
| 66534, Muscle Matrix Protein Wild Strawberry | After excluding nutrition totals, the only literal match is Strawberry, from natural Strawberry flavor. It should not become the main ingredient. |
| 77899, Calcium D-Glucarate | Both Calcium and Calcium d-Glucarate match and occur as groups. Interpret the full compound phrase before deciding whether the title names one or two ingredients. |
| 1024, Fish Oil | Groups are Calories, Fat (unspecified), EPA and DHA. Fish Oil is not a group or ingredient row here. The existing mass method picks EPA, but that does not identify the product named in the title. Preserve the unresolved source-versus-constituent distinction. |

Review flags in the output are lexical hints, not exhaustive detection of
flavors, exclusions, combinations or chemical relationships. The audit
intentionally leaves all matches as candidates rather than promoting unique
matches to confirmed identities.

## Proposed resolution policy

1. Read the product name as a phrase. Identify named ingredients, combinations,
   chemical forms, flavor descriptions and exclusions such as “sugar free”.
2. Ground each proposed ingredient in the same label's row name, group or
   recorded aliases. Preserve the exact product-name evidence, row path and
   ingredient ID. A curated alias map can extend this baseline, but record
   the alias rule used and require a compatible ingredient row on the label.
3. Select a single main ingredient when the interpreted title identifies one
   supported ingredient. For combinations, allow multiple main ingredients;
   inspect phrases such as “with” and “plus” before assigning primary versus
   supporting roles. Do not treat a flavor match as the product identity.
4. If the title is uninformative, inspect the ingredient structure and explicit
   label descriptions. A sole eligible ingredient can provide a fallback.
   Multivitamins and blends can remain formulas rather than being forced to
   one constituent. Leave genuinely unresolved cases unknown. Missing title
   matches alone are not evidence that the label has multiple main ingredients.
5. Keep mass ranking as a separate measurement. It should not break identity
   ties or substitute for interpreting the product name.

The eventual per-label output should contain `main_ingredients` as a list,
`resolution_status` (`single`, `multiple`, `formula`, `unknown`), and the
method/evidence. Each ingredient should retain its group, exact row identity,
form/source information and row path. Groups such as Blue-Green Algae or
Grape can be broader than the entity described by the title; do not erase
that specificity when aggregating claims. Candidate CSV fields in this audit
are a review input, not that final resolved output.

## Connecting purpose and risk to ingredients

Extract each claim with its own subject and provenance. The main-ingredient
mapping supplies context; it does not make every product claim a statement
about that ingredient. A multi-ingredient product's warning may concern a
supporting ingredient or the whole formula.

Keep these relationships distinct:

- `product -> has_main_ingredient -> ingredient` records the resolved identity.
- `product -> label_claims_benefit -> purpose` preserves a product-level claim.
- `ingredient -> label_claims_benefit -> purpose` is appropriate when the text
  attributes the benefit to that ingredient. Otherwise retain any proposed
  attribution separately, marked as inferred and linked to the source product.
- Risk relations should distinguish `label_warns_adverse_effect`,
  `label_warns_interaction`, `label_warns_against_use`, and
  `label_advises_consultation`, with the subject stated by the label.

For example, a direction to consult a physician during pregnancy supports a
consultation warning; it does not by itself assert an adverse effect on
pregnancy. Likewise, a label benefit claim supports `label_claims_benefit`,
not a scientifically established `effective_for` edge.

Store the DSLD label ID, verbatim evidence, source path, subject-attribution
method, and stated conditions/population with every claim. Keep confidence in
ingredient resolution separate from confidence in claim extraction. Existing
gold annotations have no ingredient-subject adjudication, so subject
attribution needs its own review before ingredient-level KG aggregation.

## Reproduction

```sh
python3 datasets/pilot/audit_title_ingredient_groups.py
```

This is an offline pilot audit requiring the local raw DSLD dump. It writes
only the new title-candidate CSV and JSON. The existing mass-based analysis,
gold claims and extraction prompt are not modified by this audit.
