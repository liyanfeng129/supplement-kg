# Recovering Purpose and Risk from DSLD labels

**Status:** exploration complete, extraction not yet run
**Date:** 2026-09-04
**Notebook:** [`notebooks/DSP_processing.ipynb`](../notebooks/DSP_processing.ipynb)

Why iDISK 2.0 leaves `Purpose` and `Risk` empty for 92,442 dietary supplement
products, whether the information exists in DSLD, and what it would take to
recover it.

---

## 1. The gap

In `datasets/iDISK2.0/Data/Entity/DSP.csv`, `Purpose` and `Risk` are populated
only for LNHPD-sourced rows. Every DSLD-only row is empty.

| group | rows | Purpose | Risk | % with Purpose |
|---|---|---|---|---|
| NHP only | 71,156 | 64,889 | 63,435 | 91.2% |
| **DSLD only** | **92,442** | **0** | **0** | **0.0%** |
| both | 208 | 208 | 188 | 100.0% |
| neither | 0 | 0 | 0 | — |

The split is perfectly clean — this is by construction, not a parsing bug.

### Confirmed two ways

**The paper states the extraction scope.** [Hou et al., JMIR 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11966073/):

> "From **LNHPD**, we extracted the dietary supplement product (DSP) name,
> **purpose**, **safety information**, and company address information"

> "From **DSLD**, we extracted the DSP name, company address information, DSI
> name, and the relationship between DSP and DSI."

**The counts match Table 1 exactly.** Non-null totals in the CSV are Purpose
65,097 and Risk 63,623 — precisely the paper's LNHPD figures.

### Why the sources differ (regulatory, not editorial)

The paper reports *what* was extracted but not *why*. The reason matters,
because it means the gap cannot be closed by re-scraping DSLD more carefully:

- **Canada (LNHPD)** licenses natural health products **pre-market**: "All
  natural health products (NHPs) sold in Canada require a product licence
  before being marketed"
  ([Health Canada](https://www.canada.ca/en/health-canada/services/drugs-health-products/natural-non-prescription/applications-submissions/product-licensing.html)).
  The licence records an authorised **recommended use or purpose**. These are
  discrete, curated, pre-approved database fields — iDISK reads them directly.

- **US (DSLD)** is a **label archive**. Under DSHEA, "FDA is not authorized to
  approve dietary supplements for safety and effectiveness before they are
  marketed," and structure/function claims "are not subject to premarket review
  and authorization by FDA"
  ([FDA](https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/label-claims-conventional-foods-and-dietary-supplements)).
  There is no authorised-indication field to publish, so DSLD transcribes what
  is printed on the package — and benefit claims live in free text.

> **Carry this downstream.** An LNHPD purpose is a *regulator-approved
> indication*. A DSLD-derived one is an *unreviewed manufacturer claim*. They
> are not the same evidentiary object and should not share a column.

---

## 2. Does DSLD actually contain the information?

**Yes, for most products.** 330 labels drawn at random from the 92,442
DSLD-only rows were read and scored — 30 read directly, 300 scored by six
parallel agents against one shared rubric.

Scoring: `2` = explicit and usable, `1` = weak/generic only, `0` = absent.

| | explicit (2) | weak only (1) | none (0) |
|---|---|---|---|
| **Purpose** | **203 (68%)** | 60 (20%) | 37 (12%) |
| **Risk** | **174 (58%)** | 109 (36%) | 17 (6%) |

Some purpose text exists on 88% of labels; some caution text on 94%.

Real examples, all from products iDISK leaves blank:

- `265051` — "SAMe supports: Joint health, Brain metabolism and mood, Liver function"
- `178419` — "Comfortable relief of occasional constipation"
- `222696` — "help you fall asleep, stay asleep, and wake up feeling refreshed"
- `241456` — "indicated to provide fish-based DHA supplementation throughout pregnancy"

Risk is often clinically specific: drug interactions (`33852`, `10383` — vitamin K
vs Coumadin/Warfarin; `63299` — warfarin; `265051` — Parkinson's/bipolar),
Prop 65 warnings, caffeine doses, age limits, allergens.

### Uncertainty on these numbers

One of the six graders (batch 3) was noticeably stricter on the explicit/weak
boundary — 48% explicit purpose vs 64–76% for the other five, mostly disputing
marketing taglines. Its labels were only marginally shorter (median 986 vs
986–1192 informative chars), so this is grader variance, not a data difference.

**Read the purpose rate as 68% ±8pp.** The 12% with *no* purpose at all was
consistent across graders and is the solid figure.

---

## 3. Why a rule-based mapping won't work

Statement types do not identify purpose. Of the 263 labels with any purpose:

| statement type hosting the purpose | share |
|---|---|
| `General Statements: All Other Content` | **58%** |
| `Formulation re: Other` | 24% |
| `Formula re: Contains` | 14% |
| everything else (incl. `Directions`) | 5% |

The dominant host is the junk-drawer bucket that also holds "Made in the USA",
"Please recycle", and "ACTUAL SIZE". A real example (`1029`, GNC Melatonin 5):

```
[General Statements: All Other Content] Made in the USA
[General Statements: All Other Content] supports restful sleep     ← the purpose
[General Statements: All Other Content] ACTUAL SIZE
```

Risk is better structured — `Precautions re: *` types genuinely exist — but is
split across several types, and allergen statements are frequently filed under
`Formula re: Contains` instead of `Precautions re: Allergies`.

**This is the empirical case for LLM extraction over a type→column mapping.**

---

## 4. Which labels can be skipped? None.

Every candidate filter leaks, tested against the 300 scored labels:

| candidate skip rule | skips | cost |
|---|---|---|
| No `Structure/Function` claim flag | 21% | **misses 14 explicit purposes** |
| < 250 informative chars | 4.7% | misses 5 purposes, 1 risk |
| < 400 chars | 9.7% | misses 10 purposes, 9 risks |
| < 600 chars | 22.7% | misses 28 purposes, 24 risks |
| No `Precautions*` type (risk only) | 7.7% | misses 2 specific risks |

The `Structure/Function` flag looks like the obvious gate and fails in both
directions: 22% of labels without it still carry an explicit purpose, and 20%
of labels with it have none.

Short labels are not safe either — length says nothing about content:

```
 43 ch  220572  SuperHair    → "Hair nutrition; Healthier, thicker, stronger"
104 ch  219387  Rush         → "Memory Focus Clarity Cognitive health support"
225 ch  26160   Antler Velvet→ "Build Your IGF Levels Naturally"
```

**And filtering saves nothing anyway.** Labels average ~260 informative tokens;
a full pass over all 92,442 rows is ~24M input tokens. Dropping the entire
<250-char tail removes 4.7% of labels but only **0.7% of the tokens** — the
skippable labels are precisely the cheap ones.

> **Decision: scan every label.** Prune boilerplate *within* each label instead.

---

## 5. Input format and cleaning

**Merge each label into one passage.** Do not split per statement:

- Purpose and risk are routinely spread across different statement types on the
  same label (`265051`: benefit in `Formulation re: Other`, contraindication in
  `Precautions`).
- Statement types are unreliable, so a per-statement call has no context.
- One passage → one output object, nothing to reconcile.

Sizes are comfortable — no chunking needed even for a 4K context:

| | median | p95 | max |
|---|---|---|---|
| tokens per label | ~342 | ~828 | ~2,200 |

### Safe to drop — verified zero benefit/risk language across all 300 labels

- `FDA Statement of Identity` (always "Dietary Supplement")
- `FDA Disclaimer Statement` (the DSHEA disclaimer)
- `General: Product/Version Code` (lot codes)

Plus: collapse whitespace, dedupe exact repeat statements (2.4% of the corpus —
DSLD files the same text under multiple types).

### NOT safe to drop — correction

An earlier version of `label_purpose_risk.py` also dropped
`Brand IP Statement(s) re: (c), (TM), (SM)` and `Seals/Symbols`. **This is
wrong.** Brand IP carries real claims in 21 statements, e.g.

```
Ocuvite Eye Vitamin Adult 50+ Formula helps replenish essential
eye protecting nutrients that you can lose as you age.*
```

Dropping it erases the *only* benefit language on 3 of 300 labels
(`1597`, `38206`, `176329`). **`engine/services/extraction/label_purpose_risk.py`
still has this bug and must be fixed before any run.**

### What cleaning buys

| variant | median tokens | saving |
|---|---|---|
| raw, everything | 404 | — |
| drop the 3 safe types | 348 | 14% |
| + dedupe | **342** | 15% |

Modest — the tokens are in the content, not the boilerplate. Aggressive
filtering is not worth the miss rate. Note the boilerplate you'd most want gone
("keep out of reach of children") sits in `Precautions re: Children`, which
cannot be dropped wholesale because that same type carries real age
restrictions.

---

## 6. Notes for a local model

1. **Keep the `[Statement Type]` prefixes.** Unreliable as a filter, useful as
   a hint — a smaller model benefits from knowing a line came from `Precautions`
   vs `General Statements`.
2. **Use constrained decoding**, not prompt-only JSON — GBNF (llama.cpp) or
   Outlines/vLLM structured output. Local models drift on format far more than
   frontier ones.
3. **Keep the evidence field.** Verifying the quoted span appears verbatim in
   the input is a free, automatic fabrication check — this matters more locally
   than with a frontier model.
4. **Consider splitting purpose and risk into two calls** on a small (7–8B)
   model. Per-label cost is trivial locally, and small models handle
   single-field extraction noticeably better than a combined schema.
5. **Grounding over recall.** If a label never states a purpose, the answer is
   null. For a knowledge base, a hallucinated indication is far worse than a
   missing one — 12% of labels genuinely have no purpose and must come back
   empty.

---

## 7. DSLD API facts

Verified live, 2026-09-04. Each of these fails silently rather than erroring:

- **Host is `https://api.ods.od.nih.gov/dsld/v9`.** The obvious
  `dsld.od.nih.gov/dsld/v9` sits behind a Cloudflare interstitial and returns
  403 to programmatic GETs. No auth needed on the working host.
- **Unknown IDs return HTTP 200 with body `[]`**, not 404 — status codes cannot
  detect a miss.
- **`DSLD_ID` cells are pipe-delimited** with a leading empty field
  (`" | 34066 | 73755"`), so `int(cell)` fails on most rows. 92,650 rows point
  at 156,957 unique labels; 31,118 rows carry more than one ID.
- **Product type, physical state, and claims are LanguaL-coded** — read
  `langualCodeDescription`, not `langualCode`.

## 8. Bulk retrieval vs. per-label fetch

Verified live, 2026-09-05. ODS publishes **no bulk export** -- the ODS site has
promised a whole-database download "as resources permit" for years, and the v9
API has no such endpoint. But `search-filter` returns label documents 1,000 at
a time, so the corpus can be paged instead of fetched one ID at a time.

- **`search-filter?q=*` reports `stats.count` = 214,780 labels.** `size` is
  honoured up to 10,000.
- **`from` is capped at 10,000** (Elasticsearch's `max_result_window`). Past
  it the API returns HTTP 200 with `hits: []` -- silent, like every other
  failure here. Paging alone therefore reaches only the first 10k labels.
- **Slicing by `date_start`/`date_end` (entry year), then `status`, then
  `product_type` where needed, partitions the database into 100 slices that
  each fit the window.** The slice counts sum to exactly 214,780, so the
  partition is provably lossless. Cost: ~371 requests, a few minutes.
- **The catch: `search-filter` returns a *summary* document, not a full
  label.** It carries `allIngredients`, claims, product type, brand and
  `netContents`, but omits `statements`, `ingredientRows`, `servingSizes`,
  `otheringredients`, `upcSku` and `targetGroups`. Since **`statements` is
  where the purpose/risk text lives**, bulk paging cannot replace the
  per-label sweep for extraction -- it is a fast census, not a substitute.
- **Filtering to iDISK barely helps, and reveals a snapshot gap.** iDISK's DSP
  rows reference 156,957 unique DSLD IDs -- 73% of the database -- so
  "download everything, then intersect" saves little over fetching the iDISK
  IDs directly. More useful is *which* labels it misses: sampling by entry
  year, iDISK covers 100% of labels entered through 2021, ~92% of 2022, and
  **0% of 2023-2025**. Its DSLD snapshot stops mid-2022, so ~58k labels from
  the last three years are absent from iDISK entirely. Only 14 of its 156,957
  IDs have no DSLD label at all.
- **Concurrency helps, but `label/{id}` IS rate limited** -- despite
  advertising no limit, and despite small samples suggesting none. Measured
  by stepping 1 -> 64 workers over disjoint IDs (2026-09-05):

  | workers | req/s | p50 | p95 | result |
  |---------|-------|-----|-----|--------|
  | 1  | 4.4  | 225ms | 245ms | all 200 |
  | 4  | 17.0 | 222ms | 242ms | all 200 |
  | 8  | 33.9 | 211ms | 232ms | all 200 -- last clean level |
  | 16 | 62.7 | 211ms | 758ms | all 200, but p95 triples |
  | 24 | 91.6 | 188ms | 768ms | 70% **429** |
  | 32+| --   | --    | --    | 100% **429** |

  The req/s above 24 is an illusion -- refusals are fast because no work is
  done. Two things make this sharper than a simple ceiling:

  - **The limit is a budget over time, not a concurrency cap.** After the
    sweep above, *every* level including 12 workers returned 429, and a
    single-threaded request did too. Concurrency alone does not describe it.
  - **It escalates.** Tripping the budget moved the endpoint from 429 to a
    bare `403` from the AWS load balancer (`Server: awselb/2.0`, no
    rate-limit headers on any response). That decayed back to 429 after
    ~3.5 min, but access was degraded for far longer than the burst.

  **Use 8 workers.** ~34 req/s with flat latency puts the 57,837 new labels
  under 30 minutes; the wall clock was never the binding constraint, and
  overrunning the budget risks the project's access rather than one run.

So: use `iter_all()` for a cheap full-corpus census (ingredients, claims,
brands), and a concurrent `label/{id}` sweep when statement text is needed.
Checkpoint per slice either way -- the silent-empty behaviour means a
truncated run looks identical to a complete one.

A verified full run: 214,780 labels in **2m48s** over 100 slices (87 MB
JSONL), against ~9h for the per-ID equivalent. Note the census is
*sequential* -- it is fast because it is ~371 requests, not because it is
parallel. Only `fetch_concurrent()` uses threads. Being ~371 requests is
also why the census never troubles the rate limit that a per-ID sweep must
respect.

### The census is also the ID discovery step

`search-filter` returns each hit's DSLD ID, which matters because nothing
else enumerates the labels iDISK has never heard of. iDISK supplies the IDs
for its own 156,957 rows; the other 57,837 exist only as the difference
between the census and iDISK. So the order is: census once (~3 min) to learn
every ID, then a concurrent `label/{id}` sweep over whichever subset needs
statement text.

The sweep itself is `dsld_bulk.sweep_labels()`, which streams: JSONL is one
record per line, so each label is appended and flushed as it arrives, and an
interrupted run keeps everything already fetched. On restart it reads back the
IDs on disk and fetches only the remainder -- including truncating a torn
final line from a `kill -9`, which would otherwise stay corrupt forever.
(`save_labels()` takes a materialised list instead, so it writes nothing until
the whole sweep succeeds -- fine for a short run, wrong for a 30-minute one.)

`harvest()` writes that split directly -- every row is tagged `in_idisk`, and
the corpus is also emitted as `-idisk.jsonl`, `-new.jsonl` and a bare
`-new-ids.txt` that feeds straight into `fetch_concurrent()`. Sweeping only
the 57,837 new labels takes **~40 min** at the observed ~23 req/s, rather
than ~3h for all 157k.

Both live in `engine/services/sources/`: `dsld_bulk.harvest()` for the census,
`DSLDClient.fetch_concurrent()` for the statement sweep. `harvest()` asserts
its own completeness by checking the fetched count against both the summed
slice counts and an unfiltered `search_count()` -- without that check a short
run is indistinguishable from a full one.

Client: [`engine/services/sources/dsld.py`](../engine/services/sources/dsld.py) absorbs all four.

---

## 9. Open questions

- **Where is the explicit/weak boundary?** This is a judgment call that directly
  sets the coverage number (the ±8pp above). Worth pinning down by hand on a
  batch before trusting any automated rate.
- **How should boilerplate risk be suppressed?** 36% of labels carry only
  generic caution text. If that lands in the `Risk` column it will dilute the
  field badly.
- **Which local model and runtime?** Determines the constrained-decoding
  mechanics and whether to split purpose/risk into separate calls.
- **Merge policy.** Confirmed: DSLD-derived values need their own provenance and
  must not be merged into LNHPD's columns. Column naming still undecided.
