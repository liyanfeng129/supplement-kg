# NHP purpose/risk → ingredient attribution in iDISK

- **Status:** proposed
- **Date:** 2026-10-16
- **Scope:** new `engine/repositories/idisk/` (`paths.py`, `loader.py`), new
  `attribute_nhp` and its types in `engine/services/ingredient_attribution/`
  (`service.py`, `schema.py`, new `excluded_dsi.txt`), ADR 0002 v5 ("Structure today"
  gains `repositories/idisk/`; no rule changes), new `tests/test_idisk_loader.py`,
  tests added to `tests/test_ingredient_attribution.py`.

Governed by ADR 0002 (latest version) and ADR 0003. Builds on spec 0009: the three
bases and their order are 0009's, and are not restated where this spec says "as 0009".

Out of scope: mapping a sentence to an iDISK relation type (`is_effective_for`,
`has_adverse_reaction`, …), deciding which sentences are claims at all, any LLM step,
DSLD-only products (spec 0009 covers them), and saving the attributions.

## Why

In iDISK 2.0 the Canadian products (those with an `NHP_ID`) are the only ones with a
`Purpose` and a `Risk`: 65,097 and 63,623 of 71,364 products. Both are text copied
from the Health Canada licence, and both sit on the **product** (`DSP.csv`). iDISK
never ties them to an ingredient: all 827 `is_effective_for` and 1,342
`has_adverse_reaction` rows in the release come from MSKCC, and the 2020 iDISK paper
(Table 3) has NHP contributing products, ingredients, `has_ingredient` and Source
Material only.

The ingredients are there, though. `dsp_dsi.csv` links 71,295 of the NHP products to
their DSIs through `has_ingredient` (316,163 links with `NHP` in `Source`), and the
links hold **medicinal ingredients only**: `Magnesium stearate` and `Hypromellose` have
no links, `Gelatin` 31. So "one link" means one active, and 36,860 products have
exactly one. For those, the licence's purpose is that ingredient's purpose, the same
argument as 0009's `single_ingredient`.

Measured on the release with a looser draft of the rules below (Common Names matched,
no punctuation cut, no exclusions), per sentence:

| basis | purpose | risk |
|---|---|---|
| `single_ingredient` | 68,499 (35%) | 107,823 (38%) |
| `named_in_claim` | 16,766 (8%) | 6,112 (2%) |
| `product_level` | 109,095 (56%) | 164,590 (59%) |

The shares match the DSLD run measured in 0009, so the same three bases are used. What
differs from DSLD is the input: here the ingredients are iDISK DSIs, not DSLD rows, and
the claims are licence text, not an extraction's claim list. Those two differences are
what this spec defines.

## In short

Every Purpose and Risk sentence already belongs to its product. This spec adds a link
to a DSI where the licence supports one. The text is first cut into sentences (a Risk
also into its sections), then each sentence goes through 0009's three questions:

1. **Does the product have exactly one DSI?** The sentence goes to it.
2. **Otherwise, does the sentence name one or more of the product's DSIs?** It goes
   to each one named.
3. **Otherwise** it stays on the product (`product_level`).

## Flow

```
caller (notebook)
  products    = idisk.loader.load_entity("DSP")          # DSP.csv rows, dicts
  ingredients = idisk.loader.load_entity("DSI")          # DSI.csv rows, dicts
  links       = idisk.loader.load_relation("dsp_dsi")    # dsp_dsi.csv rows, dicts
  attribute_nhp(products, ingredients, links) ─► list[NhpClaimAttribution]   # no I/O
```

Three reads by the caller, then one pure call, as 0009.

## The iDISK repository

`engine/repositories/idisk/`, one folder per data source (ADR 0003). No `store.py`:
the release is downloaded, never written, like the DSLD dump.

- `paths.py`: `DATA_ROOT = REPO_ROOT / "datasets" / "iDISK2.0" / "Data"`, holding
  `Entity/<name>.csv` and `Relation/<name>.csv`. Locations only.
- `loader.py`:
  - `load_entity(name)`: rows of `Entity/<name>.csv`, `name` one of `DSI`, `DSP`, `D`,
    `Dis`, `SS`.
  - `load_relation(name)`: rows of `Relation/<name>.csv`, `name` one of `dsi_d`,
    `dsi_dis`, `dsi_ss`, `dsp_dsi`.
  - Rows are `dict[str, str]` with the CSV's own column names, read as UTF-8; an empty
    cell is `""`. Returned as parsed rows rather than a dataclass, for the reason
    `dsld/raw` gives: the source keeps every column, used or not. Multi-valued cells
    (`NHP_ID`, `NHP_Name`, … with ` | `) are left as they are; splitting is the
    reader's business.
  - An unknown `name` raises `ValueError`; a missing file raises `FileNotFoundError`.

## Which products, which ingredients

- **Products:** `DSP.csv` rows with a non-empty `NHP_ID`. `Active` and `Discontinued`
  alike: a discontinued licence's text was still approved. Rows without `NHP_ID` are
  ignored.
- **A product's DSIs:** the distinct `DSI` of its `dsp_dsi` rows whose `Source`
  contains `NHP` (`NHP` or `NHP, DSLD`). Links from DSLD alone are not used: they
  describe the US label, not this licence.
- **Excluded DSIs** are listed in `excluded_dsi.txt`, one `iDISK_ID` per line with a
  `#` comment saying why, committed and diffable like `nutrition_facts.txt`. Starting
  list: `DSI000155` "acid", a concept iDISK over-merged through UMLS C0001128 "Acids"
  (its names include Taurine, Beta-alanine and L-arginine); 1,034 NHP products link to
  it, 117 as their only DSI. An excluded DSI **counts** as one of the product's DSIs (a
  product with it and one other is not single) but is **never attributed to**: a
  product whose only DSI is excluded is `product_level`.

**A product is single-ingredient** when it has exactly one DSI and that DSI is not
excluded. Forms of one ingredient are already one DSI in iDISK, so 0009's "several
actives sharing one group" case does not arise.

## Sentences

The unit attributed is a sentence, because Purpose and Risk are text blocks, not claim
lists.

1. **Purpose** is one block.
2. **Risk** is cut at `|` into sections. A section that starts with a header (text
   before the first `:`, at most 40 characters, starting with an uppercase letter,
   with no `.` in it) has that header as its `section`, and the text after the colon
   as its body. Most common in the release: `Cautions and Warnings` (61,930),
   `Contra-Indications` (897), `Known Adverse Reactions` (796). A section without a
   header has `section = None`. Empty sections are dropped.
3. **A block or body is cut into sentences** after `.`, `!` or `?` when what follows,
   after optional whitespace, is an uppercase letter or `(`. This splits the licence
   text's run-together sentences (`…fats and proteins.Helps in…`) and does not split
   `e. g. blood thinners`. Known miss: `St. John's wort` is split; accepted, since a
   wrong cut costs attribution, not correctness. Sentences are stripped; empty ones
   are dropped.
4. `index` counts a product's sentences of one kind from 0, across Risk sections.

## Attribution, per sentence

As 0009, in order:

1. **`single_ingredient`**: the product is single-ingredient. The sentence goes to
   that DSI, whatever it says.
2. **`named_in_claim`**: the sentence names one or more of the product's
   non-excluded DSIs. It goes to **every** DSI it names.
3. **`product_level`**: otherwise. `ingredients` is empty.

**A DSI's names** are its `Name`, `NHP_Name` and `DSLD_Name`, each cell split at `|`
and stripped. A name whose match key (below) is shorter than 3 characters is not
matched. **`Common Names` is not used:** it is MSKCC's
list and holds disease words; `cancer` is a common name of 6 DSIs (Protease, Coral
Calcium, Essiac, …).

**Match key.** Names and sentence pieces are compared on a key, not on
`dsld_cleaning.normalise`: lowercase; every run of characters other than letters and
digits becomes one space; a single letter followed by a space and a digit is joined
(`b 12` → `b12`); trimmed. So `Vitamin B12`, `vitamin B-12` and `Vitamin B 12` all key
to `vitamin b12`, and `Coenzyme Q-10` to `coenzyme q10`. 0009 can reuse `normalise`
because its claims were copied from passages that `normalise` rendered; licence text
was not, and `normalise` drops quantities, which turns `Vitamin B 12` into `vitamin b`.
Known miss: `omega-3` (`omega 3`) and `omega3` keep different keys.

**A name matches** when its key occurs as whole words in the key of a sentence piece.
The sentence is cut into pieces at `,` `;` `:` `(` `)` `.` `!` `?` first, so a name
never **crosses punctuation**: `Multi-vitamin, a factor in…` would otherwise key to
`multi vitamin a factor`, which names Vitamin A; 31 Purpose fields have that pattern.
A hyphen is not a cut.

## Interface

```python
# engine/services/ingredient_attribution/schema.py (beside 0009's types)
@dataclass(frozen=True)
class DsiRef:
    dsi_id: str              # iDISK_ID, e.g. "DSI000002"
    name: str                # DSI.csv Name

@dataclass(frozen=True)
class NhpClaimAttribution:
    dsp_id: str              # the product's iDISK_ID
    kind: str                # "purpose" | "risk"
    section: str | None      # Risk section header; None for purpose or a headerless section
    index: int               # position among the product's sentences of this kind
    text: str                # the sentence as printed
    basis: str               # "single_ingredient" | "named_in_claim" | "product_level"
    ingredients: tuple[DsiRef, ...]   # empty for product_level; in ingredients order

    def to_dict(self) -> dict: ...
```

```python
# engine/services/ingredient_attribution/service.py
def attribute_nhp(products: list[dict], ingredients: list[dict],
                  links: list[dict]) -> list[NhpClaimAttribution]:
    """Attribute every Purpose and Risk sentence of every NHP product to DSIs."""
```

- Inputs are the loader's rows. A link whose `DSI` is not in `ingredients` raises
  `KeyError` (none does in the release); links to products not in `products` are
  ignored.
- Output is in `products` order, then purpose before risk, then sentence order.
- No I/O beyond reading `excluded_dsi.txt` once; no model, no randomness. Changes no
  input.

### Where each step lives

| Function | Does |
|---|---|
| `load_entity(name)`, `load_relation(name)` (public, `repositories/idisk/loader.py`) | read one CSV of the release into rows |
| `attribute_nhp(products, ingredients, links)` (public, `service.py`) | loads `excluded_dsi.txt`; indexes DSIs and their names, and each product's NHP DSIs; per product cuts Purpose and Risk into sentences, decides single-ingredient, then per sentence applies the three bases in order |

No other function; the sentence cutting, match key and matching stay inline in
`attribute_nhp` (CODING_STANDARDS rule 1).

## Acceptance

1. `load_entity("DSP")` and `load_relation("dsp_dsi")` return rows with the CSV's
   columns, `""` for empty cells; `load_entity("Foo")` raises `ValueError`. Tests use a
   patched `paths.DATA_ROOT`.
2. A product with one NHP DSI: every Purpose and Risk sentence is `single_ingredient`
   to it.
3. A product with two NHP DSIs: a sentence naming one is `named_in_claim` to it alone;
   a sentence naming both goes to both; a sentence naming none is `product_level`.
4. A DSLD-only link does not count: a product with one NHP DSI and one DSLD-only DSI is
   single-ingredient.
5. A product whose only DSI is excluded: all `product_level`. A product with an
   excluded DSI and one other: not single; a sentence can be `named_in_claim` to the
   other only.
6. Risk `Cautions and Warnings: A. B. | Contra-Indications: C. | ` gives three
   sentences, sections `Cautions and Warnings`, `Cautions and Warnings`,
   `Contra-Indications`, indexes 0, 1, 2.
7. `…fats and proteins.Helps in…` is two sentences; `(e. g. blood thinners)` is not cut.
8. `Multi-vitamin, a factor in good health` does not name Vitamin A;
   `Helps to prevent vitamin B-12 deficiency` and `… Vitamin B 12 …` both name a DSI
   whose name is `Vitamin B12`; `Iron` does not match `ironman`; a two-character key
   never matches.
9. A `Common Names` entry never matches; an `NHP_Name` entry does.
10. Products without `NHP_ID` produce nothing; inputs are unchanged after the call.
11. **On the full release**, in a notebook: counts per basis and kind, reported beside
    the table in [Why](#why). Then, in conversation, read 30 random `named_in_claim`
    and 30 random `single_ingredient` attributions against the product's ingredient
    list, and report wrong attributions. The user confirms `excluded_dsi.txt` and the
    open questions from that review.

## Open questions

- **Q-A: Generic risks on single-ingredient products.** The rule is decided in 0009
  Q-A (2026-10-16): a sentence naming a condition of the user (pregnancy,
  breastfeeding, age, a medication, a named disease) is a risk and is attributed; one
  with no condition, or whose only condition is about the product's effect ("If
  symptoms persist or worsen, consult your health care practitioner"), is
  boilerplate. **Still open here:** in 0009 the renderer v4 checklist removes that
  boilerplate before extraction; licence text never passes through it, and its
  Canadian wording would not match the checklist anyway. So this spec, as written,
  attributes the boilerplate too, and most Risk text is `Cautions and Warnings`
  boilerplate. Needed: a checklist of licence boilerplate sentences, built like
  `boilerplate-candidates.md` (most frequent Risk sentences, ticked by the user),
  and a decision on whether `attribute_nhp` drops them or marks them. `section` is
  kept so a section can also be judged whole. Separately, the iDISK paper has no
  relation for a precaution; its adverse relations point at effects
  (`has_adverse_effect_on` an organ system, `has_adverse_reaction` a symptom), and
  its free-text Safety attribute is where a precaution sits.
- **Q-B: Homeopathic products.** 4,332 products say "homeopathic" in their Purpose,
  often only "Homeopathic Medicine", which is not a purpose; their ingredients are
  dilutions. Included by default; dropping them is a filter on `products` by the
  caller.
- **Q-C: Naming is not being about.** "Consult a health care practitioner prior to use
  if you have an iron deficiency" names Iron; "Helps in the absorption and use of
  calcium and phosphorus", a vitamin D claim, names Phosphorus. Same rule and same
  weakness as 0009's `named_in_claim`; acceptance 11 measures how often.
- **Q-D: Provenance.** These sentences are regulator-approved licence text; 0009's are
  manufacturer claims extracted by a model. Whatever saves either keeps that apart, as
  the iDISK paper keeps sources on every atom.
- **Q-E: Saving**, as 0009 Q-C.
