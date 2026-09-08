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
- **`label/{id}` is capped at 1,000 requests/hour/IP** -- documented in the
  DSLD API guide (dsld.od.nih.gov/api-guide). That is **0.28 req/s, one
  request every 3.6s**, and it is the binding constraint on every per-ID
  plan. No response carries a rate-limit header, so nothing at runtime
  reveals it; it has to be read from the guide.

  A concurrency benchmark run before finding that number is a cautionary
  tale worth recording. Stepping 1 -> 64 workers looked clean through 8
  workers at ~34 req/s -- **122x the documented limit** -- because a short
  burst *spends the hourly budget* rather than being refused up front. Past
  that the endpoint returned 429 at every concurrency including 1, then
  escalated to a bare `403` from the load balancer (`Server: awselb/2.0`)
  that took minutes to decay. **A passing benchmark proves nothing against a
  budget limit; it just drains it faster.** The 240-request test alone was
  24% of an hour's allowance.

  The consequence for planning: at 1,000/h a per-ID sweep is **days, not
  hours** -- ~58h for the 57,837 labels new since iDISK, ~157h for iDISK's
  own set, ~215h for the whole database. The "under 30 minutes" figure from
  the benchmark was never achievable.

### The official full download supersedes both

ODS does publish a whole-database export after all. The copy in
`datasets/DSLD-full-database-JSON/` holds **214,780 JSON files** -- one per
label, bucketed 1,000 to a directory -- which matches the API's own count
exactly. Its `ReadMe.csv` dates the download to **2026-01-10**; the older
`2025-09-25` path inside each file's `src` field is ODS's internal build
stamp, not the download date.

Each file is the **full** label document: the same key set as `label/{id}`,
including the `statements`, `ingredientRows`, `servingSizes`, `contacts` and
`otheringredients` that `search-filter` omits. Verified by fetching 8 random
labels spanning entry years 2012-2023 and comparing to the live API -- all 8
identical, field for field.

The encoding is clean UTF-8 (`\xc2\xae` for `®`). An earlier note here
claimed the dump was mojibaked; that was wrong. The corruption was introduced
by pasting a file's contents through a lossy channel, and a scan of all
214,780 files found **zero** affected -- 78.6% contain non-ASCII text, none of
it damaged. No repair pass is needed, and adding one would corrupt genuinely
clean text.

This removes the rate limit from the critical path entirely: with the export
in hand no per-ID sweep is needed for statement text. The API is now only for
refreshing individual labels or picking up entries added since 2026-01-10.

So the census is not merely the faster option, it is the only per-label
route that fits the budget: `iter_all()` covers all 214,780 labels in ~371
requests, comfortably inside a single hour's allowance, while the equivalent
per-ID sweep would take nine days. Use `label/{id}` only for fields the
summary documents genuinely lack -- above all the statement text -- and
scope that sweep as narrowly as the question allows.
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

---

## 10. Pilot implementation

Built 2026-09-08. The pipeline below runs end to end, gold set included.

### What each piece does

| Module | Role |
|---|---|
| [`sources/dsld.py`](../engine/services/sources/dsld.py) `load_label` | Reads one label from the local dump into the same `Label` the API client returns |
| [`extraction/pilot_sample.py`](../engine/services/extraction/pilot_sample.py) | Draws the DSLD-only frame at ID grain; `build_pilot` writes the flattened passages |
| [`extraction/label_purpose_risk.py`](../engine/services/extraction/label_purpose_risk.py) | Renders the passage, calls the model, grounds and scores the result |
| [`extraction/gold_set.py`](../engine/services/extraction/gold_set.py) | Runs the frontier model over 100 pilot labels into a committed reference set |

### Section 5's filter bug is fixed

`label_text` had two filters: `SKIP_TYPES` and an undocumented
`INFORMATIVE_PREFIXES` allow-list. Both excluded `Brand IP` and
`Seals/Symbols`, so labels `38206` and `176329` reached the model with their
only benefit claim already removed and could only come back empty. Now only
the three verified-safe types are dropped, and the whitespace collapse and
statement dedupe this section specified are implemented.

### Frame and coverage — section 5's open risk, measured

The concern was that iDISK might reference IDs the dump lacks, in numbers that
would change what extending to the whole corpus means. It does not:

| | |
|---|---|
| DSLD-only DSP rows | 92,442 |
| unique DSLD IDs behind them | 156,548 |
| present in the 2025-09-25 dump | 156,534 |
| **missing** | **14** (listed in the manifest) |

Row counts reproduce section 1's table exactly. The 208 `both` rows are
excluded from the frame and held back as the independent validation stage.

Passage sizes come out at 1,648 chars median and 8,960 max — near this
document's ~342 and ~2,200 token estimates, confirming no chunking is needed.

### Grounding is now enforced, not just recorded

Section 6.3 called the evidence span a free fabrication check; it is now wired
in. Every entry records whether its evidence occurs in the passage, compared
whitespace-insensitively so a span quoted across a line break is not failed for
formatting.

That check **caps the confidence score**. The model rates its own certainty,
which a confident hallucination scores as highly as a correct reading; the
substring check cannot be talked up, so it bounds the score it cannot set. An
empty list scores 0 regardless. The ungrounded count is reported in every
manifest.

**The known weakness, restated:** for purpose, `text` and `evidence` are
near-identical and the check is strong. For risk, DSLD packs several cautions
into one `notes` block with no substring isolating one, so `evidence` is the
whole block. The check then proves *the block is genuine*, not *the claim drawn
from it is right*. It catches wholesale fabrication, not misreading. Bounded
and accepted, not solved.

### Closes one open question from section 9

**Merge policy — column naming.** DSLD-derived values are written as
`Purpose_dsld` / `Risk_dsld` with a `dsld-label-llm` source marker, never into
the `Purpose` / `Risk` columns holding LNHPD's regulator-approved indications.

Boilerplate suppression remains open. The per-entry `text` field is where such
a pass would act, but the pass itself is still undesigned — which is why the
list shape matters: collapsing entries into one string later is trivial,
splitting a joined string back apart means re-running.

### The gold set exists

`datasets/pilot/gold-100.jsonl`, 100 labels, **hand-labelled by reading each
rendered passage** rather than by a scripted API sweep. 146 purpose entries and
190 risk entries; all 336 evidence spans occur verbatim in their passage.

The grounding check proved itself during the labelling: it caught a dropped
word in a transcribed allergen list on `18162` — the exact class of error it
exists to find, made by hand rather than by a model.

**28 of 100 labels state no purpose**, against the ~12% estimated in section 2.
That gap matters for what the set is good for. The empty ones are the products
where supplying outside knowledge is most tempting — white willow bark, ginkgo,
dong quai, boswellia, DHEA — and none of their labels make a claim. `13504`
says only that Dong Quai is "highly treasured"; `5174` (Policosanol) states
nothing at all, while `7991`, also a policosanol product, does claim
cholesterol support. The split is label-driven, not ingredient-driven, so a
local model that pattern-matches on ingredient name will fail these 28 visibly.
That makes the set a harder target than the estimate implied, and a better one.

`gold_set.build` remains the scripted path for regenerating or extending the
set through the API. It is verified against a stubbed client but has never been
run for real: `.env` holds only NCBI credentials, and there is no
`ANTHROPIC_API_KEY` on this machine.

Still deferred, and still for the reason section 9 gives: which local model and
quant. That now waits on downloading weights, not on having something to score
against.
