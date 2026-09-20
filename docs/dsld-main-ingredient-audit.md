# Main ingredient from the 1,000 audited DSLD labels

The result is in `datasets/pilot/dsld-main-ingredients-1000.csv`. The companion
`.audit.json` records source paths, unresolved structures, excluded aggregates,
and comparisons for each label. `notebooks/05_dsld_main_ingredient.ipynb`
contains one processing step per cell, without a wrapper cleaning function.
It accepts an existing `raw_labels`; otherwise it loads the audited sample.

The source JSONL supplies only the 1,000 IDs. All ingredient names and amounts
come from the original DSLD JSON records; purpose/risk gold annotations are
not consulted or changed.

| Result | Labels |
|---|---:|
| One eligible non-blend ingredient with no nested components | 319 |
| Unique largest mass with complete comparable amounts | 191 |
| Largest among quantified ingredients only | 291 |
| No comparable result | 155 |
| Tie or disagreement between serving bases | 44 |

`main_ingredient` is populated for the first two groups (510 labels).
`largest_among_quantified` is populated only for the partial group. It must
not be promoted to a confirmed main ingredient. Empty CSV fields mean unknown.

## Operational rules

- The scope is named supplement ingredient rows, excluding `otheringredients`
  such as excipients, solvents and capsule material. This is a declared-mass
  ranking, not a judgement of pharmacological importance or effectiveness.
- Remove an explicit list of generic nutrition aggregates, including calories,
  total carbohydrates, generic protein/fiber, sugars and fat totals. This is an
  exact-name list, not a broad keyword filter. Named source ingredients such as
  whey protein isolate are retained. Excluded rows and their subtrees are logged.
- Traverse components of explicit blends (category `blend` or the word
  `blend` in the row name). Never rank the parent blend total against its children.
  A blend without listed components prevents confirmation.
- Non-blend nesting is ambiguous: it can describe a chemical source compound,
  a constituent, or a concentration. Mark it unresolved rather than automatically
  treating parent and child as independent ingredients. This is a conservative
  implementation limit, not proof that the label is intrinsically unresolvable.
- A sole eligible, non-nested, non-blend ingredient can be named without an amount.
  “Sole eligible” is measured after aggregate removal, so it differs from simply
  counting raw top-level rows.
- Otherwise require exact `=` mass amounts with recognized units. Preserve
  declared zero masses; reject NP/missing units, IU/CFU/activity units and bounds.
  Convert grams, milligrams, micrograms and nanograms to milligrams.
- Compare only the same serving reference **and actual serving quantity/unit**.
  Fill a missing quantity only from a compatible fixed serving definition.
  Never compare a one-capsule amount with a two-capsule amount merely because
  their `servingSizeOrder` is identical.
- Confirm a quantitative winner only when every comparison has all eligible
  ingredients, no unresolved structures, no conflicting duplicate amounts, one
  positive winner, and the same winning row across serving bases. Partial
  comparisons have a separate column; ties and disagreement remain unknown.

These are auditable structural heuristics, not human-adjudicated ingredient
identities. Different chemical forms and equivalent names are not merged.
Aggregate terminology and ambiguous nested structures may require further
curation. No total-weight percentage is inferred by summing rows.

## Early exits and avoided work

The notebook now resolves cases in this order:

| Processing path | Labels | Work performed |
|---|---:|---|
| Sole eligible ingredient | 319 | No quantity records read |
| No eligible ingredient | 48 | No quantity records read |
| No usable exact mass | 101 | Availability check only |
| One positive usable amount | 61 | Availability check; no conversion or ranking |
| Remaining comparisons | 471 | Normalize serving bases and masses; compare candidates |

Thus 529 labels skip conversion and ranking. Of the 291 partial leaders, only
61 are handled by the single-amount shortcut: identifying the largest of
several quantified candidates still requires comparison. Repeated quantity
records go through serving/conflict checks rather than being silently reduced
to one. Groups with one quantified candidate do not call a maximum search.

This is a reduction in work, not a measured wall-clock speedup. The availability
pass scans 6,575 quantity records; the remaining comparison pass converts 3,930
mass records and ranks 479 serving groups. All columns in all 1,000 result rows
were checked against the pre-optimization CSV and are unchanged. The audit JSON
records which processing path produced each result and the evidence for the
single-amount shortcut.

## Validation

Run `python3 datasets/pilot/find_main_ingredients.py` to reproduce the analysis
in memory and print counts. All ten notebook code cells were executed and
their 1,000 names matched the CSV. `tests/test_main_ingredients.py` checks 13
synthetic cases covering mass conversions, blends, missing and incompatible
amounts, bounds, ties, conflicting serving winners and chemical-form nesting.
It now also checks conflicting duplicate amounts (14 selection cases total),
and explicitly verifies that a sole ingredient's quantities are never read and
the single-amount shortcut performs no conversion or ranking.
No production extraction code or original loader-tour cells were changed.
