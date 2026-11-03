# Claim → ingredient attribution: which claims the label ties to an ingredient

- **Status:** proposed
- **Date:** 2026-10-13
- **Scope:** new `engine/services/ingredient_attribution/` (`service.py`, `schema.py`,
  `nutrition_facts.txt`), `engine/services/dsld_cleaning/service.py` (module-level
  `normalise`), new `tests/test_ingredient_attribution.py`.

Governed by ADR 0002 (latest version) and ADR 0003. Reads what ADR 0001's extraction
writes; changes nothing about the extraction, its prompt or its `config_hash`.

Out of scope: saving the attributions (see [Open questions](#open-questions)), mapping
ingredients to iDISK concepts, any LLM step, and attributing product-level claims.

## Why

A purpose/risk row belongs to a **product** (`dsld_id`). The knowledge graph wants
claims on **ingredients** too. The raw DSLD record lists the product's ingredients
(`ingredientRows`), but a claim says which ingredient it is about only sometimes.
Measured on the 4,044 claims of run `gold1000-r4-2610130950-0`:

| basis | purpose | risk |
|---|---|---|
| `single_ingredient`: the label has one active ingredient | 669 (28%) | 568 (34%) |
| `named_in_claim`: several actives, the claim names one or more | 114 (5%) | 109 (6%) |
| `product_level`: several actives, the claim names none | 1,571 (67%) | 1,013 (60%) |

For the first two the label itself says which ingredient: there is only one, or the
claim names it. Those are attributed by rule here. For the third the label does not
say, and an attribution would come from a model's general knowledge rather than the
label, so they stay on the product. The basis travels with every attribution, so a
reader can always tell the two apart.

### Why not iDISK's `has_ingredient`

iDISK 2.0 links products to ingredients (`dsp_dsi.csv`), but it cannot stand in for
`ingredientRows` on DSLD labels ([spot check](../dsp-dsi-dsld-spot-check.md),
2026-09-19):

- **Coverage.** Only 2,156 of its 317,062 links lead to a DSLD label, covering 1,477
  label ids, under 1% of the 214,780 in the full run.
- **Completeness.** The DSLD-sourced links are often incomplete: "Resveratrol 500 mg"
  links only Grape, though the label also lists Japanese knotweed and quercetin; a
  "Best L-Tryptophan 500 mg" product links a vitamin B6 form but not L-tryptophan. The
  NHP-sourced links in the same check matched the label (8 selected products, not a
  random sample).
- **It says what a product contains, not what a claim is about.** At best it gives
  the `single_ingredient` basis; it does not settle a `product_level` claim.

For NHP products the links are the right source: they hold medicinal ingredients only,
so a product with exactly one link is a single-ingredient product (36,860 of them).
Spec [0010](0010-nhp-claim-ingredient-attribution.md) attributes NHP Purpose and Risk
that way.

## In short

Every claim already belongs to its product: the purpose/risk row is keyed by
`dsld_id`. This spec only adds a link to an ingredient where the label supports one.
Per claim, three questions in order:

1. **Is there only one ingredient on the label?** Then the claim, purpose or risk, goes
   to it. "Only one" means one, not "the main one": after dropping Nutrition Facts
   rows (Calories, Fat, Sugar, …) and replacing blends by what they contain, exactly
   one active is left, or several that share one group (two forms of Vitamin C). A
   botanical with Vitamin C and Zinc has no single ingredient, however obvious the
   botanical's role.
2. **Otherwise, does the claim name an active?** Anywhere in its text, as whole words,
   by the active's name, group or an alternate name, after the same normalisation the
   passage went through ("supports healthy vitamin b 12 levels" names `Vitamin B-12`).
   The claim goes to every active it names, possibly several.
3. **Otherwise** the claim gets no ingredient and stays on the product only
   (`product_level`): 60–67% of claims in the measurement above.

## Flow

```
caller (notebook)
  rows    = runs.loader.read_rows(run)                       # purpose/risk rows, dicts
  records = raw.loader.load_raw([r["dsld_id"] for r in rows]) # raw DSLD records, dicts
  attribute(rows, records) ─► list[ClaimAttribution]          # no I/O
```

Two reads by the caller, then one pure call (ADR 0003 rule 2: no repository reads
another's data).

## Active ingredients of a label

The ingredients a claim can be attributed to. Read from the raw record's
`ingredientRows`, walking `nestedRows`:

1. **Nutrition Facts rows are dropped, with their nested rows.** A row is Nutrition
   Facts when its `ingredientGroup` is in `nutrition_facts.txt`, one group per line,
   committed and diffable like `dsld_cleaning/boilerplate.txt`. The `category` field
   cannot do this: `Sodium` and `Calcium` are `mineral` whether they are a Nutrition
   Facts line or the product's point. Starting list, from the full-dump census:
   `Calories`, `Carbohydrate`, `Carbohydrate (unspecified)`, `Fat (unspecified)`,
   `Saturated fat`, `Trans fat`, `Trans fat (unspecified)`, `Polyunsaturated Fat`,
   `Monounsaturated Fat`, `Unsaturated Fat`, `Cholesterol`, `Sugar`,
   `Sugar (unspecified)`, `Sugar Alcohol`, `Fiber (unspecified)`, `Protein`,
   `Protein (unspecified)`. `Sodium` is open (Q-B).
2. **A blend is replaced by its nested rows.** A blend is a row whose `category` is
   `blend` or whose `ingredientGroup` contains `Blend`. Its nested rows go through the
   same rules, at any depth. A blend **with no nested rows** stays as one active,
   with `group = None`: an ingredient that exists but cannot be named.
   **A `Header` row** (6,700 in the dump, a section title such as "Proprietary
   Formula") is replaced by its nested rows the same way, but with no nested rows it
   is dropped: a title is not an ingredient.
3. **Any other row is one active; its nested rows are ignored.** Nested rows under a
   non-blend are forms or sources of it (Folate → folic acid, Fish Oil → EPA, DHA).
4. **An active's group** is its `ingredientGroup`, or `None` when that is missing
   (0.07% of rows; they are kept, not dropped).
5. `otheringredients` (excipients) are never actives.

**A label is single-ingredient** when it has exactly one active and that active has a
group, or several actives that all share one non-`None` group (e.g. Vitamin C as
ascorbic acid and as sodium ascorbate).

## Attribution, per claim

Only `ok` rows are read: a failed row has no claims. Each claim of each row gets one
`ClaimAttribution`, decided in this order:

1. **`single_ingredient`**: the label is single-ingredient. The claim goes to that
   ingredient, whatever it says.
2. **`named_in_claim`**: the claim text contains, as whole words, the normalised
   `name`, `ingredientGroup` or any `alternateNames` entry of one or more actives.
   The claim goes to **every** active it names. Names are normalised with
   `dsld_cleaning.normalise`, the same function that rendered the v4 passage the
   claim was copied from, so `Vitamin B-12` and the claim's `vitamin b 12` agree. A
   normalised name shorter than 3 characters, or a group `None`, is not matched.
3. **`product_level`**: otherwise, including a label with no actives (12 in the full
   dump). `ingredients` is empty.

A `named_in_claim` match on an active whose group is `None` still attributes, with
`group = None`; it carries the `ingredient_id` and `name`.

## Interface

```python
# engine/services/ingredient_attribution/schema.py
@dataclass(frozen=True)
class IngredientRef:
    group: str | None        # DSLD ingredientGroup: the knowledge-graph ingredient node
    ingredient_id: int | None
    name: str                # as printed on the label

@dataclass(frozen=True)
class ClaimAttribution:
    dsld_id: int
    kind: str                # "purpose" | "risk"
    index: int               # position of the claim in the row's `purpose` / `risk` list
    text: str
    basis: str               # "single_ingredient" | "named_in_claim" | "product_level"
    ingredients: tuple[IngredientRef, ...]   # empty for product_level

    def to_dict(self) -> dict: ...
```

```python
# engine/services/ingredient_attribution/service.py
def attribute(rows: list[dict], records: list[dict]) -> list[ClaimAttribution]:
    """Attribute every claim of every `ok` row to ingredients, by the rules above."""
```

- `rows` are purpose/risk run rows; `records` are raw DSLD records. Joined by
  `dsld_id` / `id`. A row with no record raises `KeyError`; extra records are ignored.
- Output is in row order, then `purpose` before `risk`, then claim order.
- No I/O, no model, no randomness: the same input gives the same output.
- Changes neither input.

### Where each step lives

| Function | Does |
|---|---|
| `attribute(rows, records)` (public, `service.py`) | loads `nutrition_facts.txt` once; indexes records by id; per `ok` row walks the record's `ingredientRows` to its actives (rules 1–5 inline), decides single-ingredient, then per claim applies the three bases in order |
| `normalise(text)` (public, module level in `dsld_cleaning/service.py`) | the existing `DsldCleaningService._normalise`, exposed like `clean`, `flatten` and `cluster` so attribution matches names with the renderer's own normalisation, not a copy of it |

No other function. The active-ingredient walk stays inline in `attribute`
(CODING_STANDARDS rule 1).

## Acceptance

1. A label with one active row: every claim is `single_ingredient` to it.
2. A label whose actives are `Calories`, `Fat (unspecified)` (with nested
   `Saturated fat`) and one botanical: single-ingredient, to the botanical.
3. A label with one `Proprietary Blend` row whose nested rows are two botanicals: not
   single; a claim naming one botanical is `named_in_claim` to it alone.
4. A label with one blend row and no nested rows: not single-ingredient; its claims
   are `product_level` unless one names the blend's printed `name`.
5. Two actives with the same group (two forms of Vitamin C): single-ingredient.
6. A claim naming two actives is attributed to both; a claim naming none is
   `product_level` with empty `ingredients`.
7. Name matching is whole-word on normalised text: `Vitamin B-12` matches
   `supports vitamin b 12 levels`; `Iron` does not match `ironman formula`; a
   two-letter name never matches.
8. An `alternateNames` entry matches like the name.
9. A non-`ok` row produces no attributions; a label with no actives gives
   `product_level` for all its claims.
10. A row with no matching record raises `KeyError`; the inputs are unchanged after
    the call.
11. **On `gold1000-r4-2610130950-0`**, in a notebook: the counts per basis and kind are
    reported beside the table in [Why](#why) (they will move, because the rules here
    are tighter than the measurement's). Then, in conversation, read 30 random
    `named_in_claim` and 30 random `single_ingredient` attributions against their
    passage and ingredient list, and report wrong attributions. The user confirms
    `nutrition_facts.txt` and the open questions from that review.

## Open questions

- **Q-A: Generic risks on single-ingredient labels. Decided 2026-10-16: attribute
  them.** A risk that names a condition of the user (pregnancy, nursing, age, a
  medication, a named disease) is a real risk: "consult your physician if pregnant"
  on a melatonin-only label is a melatonin risk. A sentence with no such condition
  ("consult your healthcare practitioner prior to use") is boilerplate, and so is one
  whose only condition is about the product's effect ("if symptoms persist or
  worsen", "if adverse reactions occur"). The renderer v4 checklist already draws
  this line (the bare "consult…" phrases are on it, no pregnancy or nursing line is),
  so boilerplate never reaches a claim and `single_ingredient` applies to risks as to
  purposes. A catch-all list ("if pregnant, nursing, taking any medication or have a
  medical condition") is one risk; separating its generic clauses belongs to mapping
  risks to relations, out of scope here. Pregnancy warnings sit on thousands of
  labels, so whatever ranks risk edges later weighs them by how many labels repeat
  them.
- **Q-B: Sodium.** Usually a Nutrition Facts line, sometimes the product (electrolyte
  mixes). DSLD also groups forms under `Sodium` whose point is another nutrient
  (`Sodium Molybdate`, `Sodium Selenate`, `Sodium Ascorbate`); nested under their
  parent they are ignored anyway (rule 3), but a top-level one is a real active.
  Sodium bicarbonate is mostly its own group, `Sodium Bicarbonate` (sample of dump
  dirs `000000`–`009999`), so this list does not touch it. Left **out** of
  `nutrition_facts.txt`, so a label
  listing Sodium plus one botanical is not single-ingredient: it costs coverage, not
  correctness. Calcium is left out for the same reason, more strongly.
- **Q-C: Saving.** Like spec 0006, this spec returns values and saves nothing. When
  the full run is attributed, the output is a derived file of one run; where it lives
  (inside the run directory beside `purpose_risk.jsonl`, or its own data source) is
  decided by whoever reads it back.
- **Q-D: A check set for ingredient matching.** Mapping a DSLD ingredient to an iDISK
  DSI is out of scope here, but the 1,477 DSLD labels iDISK already links (see
  [Why not iDISK's `has_ingredient`](#why-not-idisks-has_ingredient)) give known
  label–DSI pairs to measure a matcher against, the NHP-sourced ones first, as the
  DSLD-sourced ones are incomplete.
