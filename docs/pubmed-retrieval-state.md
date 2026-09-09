# PubMed abstract retrieval — state at pause

**Status:** paused 2026-09-02, focus moved to DSLD purpose/risk extraction.
**Branch:** `experiment/pubmed-retrieval` (the same branch the DSLD work later
continued on — the PubMed commits are `b5efea0` and `8d029fd`, everything after
`2636833` is DSLD/iDISK).

This document exists because the state of this work lived only in commit messages
and notebook cells. Nothing here is a plan; it records what was built, what was
measured, and what was left open.

## What was built

`engine/services/sources/pubmed.py` (407 lines) — retrieval only. No extraction.

- **`QuerySpec`** — declarative description of one search. Substance, synonyms and
  health terms are *separately addressable fields*, not one opaque query string.
  Each filter is an independent boolean (`interventional_only`, `humans_only`,
  `exclude_non_research`, `require_abstract`, `english_only`, date range).
  `build()` renders it to PubMed syntax, expanding every term across both `[tiab]`
  and `[mh]`.
- **`PubMedClient`** — E-utilities wrapper. Throttles to 0.15s with an API key /
  0.4s without (NCBI publishes 10/s and 3/s; the code leaves headroom deliberately,
  because a 429 mid-run costs more than the added latency). Retries 429/5xx with
  exponential backoff. Uses the history server (`usehistory=y`) so paging does not
  re-run the search.
- **`parse_article`** — keeps structured-abstract section labels
  (`OBJECTIVE:`/`METHODS:`/`RESULTS:`/`CONCLUSIONS:`) inline in the abstract text,
  because those labels carry PICO signal. Also captures publication types and MeSH
  descriptors.
- **`run_query` / `new_run_dir` / `save_run`** — every execution writes an
  immutable timestamped directory.

**Retrieval is anchored to claims, not to supplements.** Every abstract carries the
`claim_id` that surfaced it. The point: downstream PICO extraction can be checked
against the claim actually searched for, which separates extraction failures from
retrieval failures.

### Run directory shape

```
data/pubmed/runs/<timestamp>-<label>/
  run.json              # committed — params, per-query hit counts, corpus_hash
  manifests/<claim>.json# committed — query, translation, sorted PMIDs, pmid_hash
  raw/<claim>.jsonl     # gitignored — the abstracts themselves
```

`corpus_hash` changes if and only if the retrieved PMID set changed, so re-running
after a filter change tells you whether the change actually moved anything.

## What was measured

Two runs were executed against live PubMed on 2026-09-02. Both raw payloads are
still present on disk (they are gitignored, not deleted — 688K and 164K).

| Run | Claims | Total hits | Fetched | Unique PMIDs |
|---|---|---|---|---|
| `20260902T161349Z-seed8-strict` | 8 | 5,015 | 200 | 199 |
| `20260902T161418Z-seed2-relaxed` | 2 | 669 | 50 | 50 |

**The headline number: the study-type filter keeps only 18.7% of hits** (5,015 of
26,810) across the eight seed claims. The denominator is those same eight queries
with `interventional_only` flipped off and every other filter held constant, so it
isolates that one lever. It was left on by default, but the notebook quantifies the
cost rather than assuming it.

**What the filter buys, from the two runs' publication-type mix:**

- strict (200 records): 99 Randomized Controlled Trial, 89 Systematic Review,
  42 Meta-Analysis. Years 2013–2026.
- relaxed (50 records): 20 Review, only 4 Randomized Controlled Trial. Years 2023–2027.

So the discard is not indiscriminate. Without the filter, the top-N by PubMed's
default ordering fills with narrative reviews — the least extractable material.
That is an argument for keeping the filter, or for pairing any relaxation with an
explicit sort/rank strategy rather than taking the first 25.

Hit counts are very uneven across claims: Vitamin D×bone 2,036 and probiotics×IBS
1,621, against Zinc×cold 101 and Serenoa×BPH 73. A fixed `limit_per_claim` therefore
samples the big claims far more shallowly than the small ones.

### The eight seed claims

Serenoa repens×BPH · Melatonin×sleep · DHA×cognition · Vitamin D×bone ·
Curcuma longa×inflammation · Probiotics×IBS · Zinc×cold · Ginkgo biloba×cognition.

Hand-written on purpose. The EU register was *not* used to drive retrieval because
its `health_relation` column is null on 433 of 3,342 rows, and its `claim` text is
regulatory prose ("contributes to the normal function of the immune system") rather
than search vocabulary. Mapping that prose to MeSH is its own task; the seed set
isolates the retrieval question from the claim-normalisation question.

## What was never built

- **No PICO extraction.** This was always the stated downstream consumer, and no
  part of it exists. `parse_article` preserving section labels is the only thing
  done in anticipation of it.
- **No ontology matching.**
- **No use of the retrieved corpus.** The 250 abstracts have never been read by
  anything but the notebook's eyeball cells.

## Open questions (from `notebooks/01_pubmed_retrieval.ipynb`, section 9)

1. **Multi-substance claims.** `_or_block` ORs a substance with its synonyms, so
   *"Calcium and vitamin D"* retrieves calcium **or** vitamin D, never the
   combination. Many EU register claims have this form. Needs an AND-group
   representation in `QuerySpec` — this is a real modelling gap, not a tuning knob.
2. **Claim text → search vocabulary.** Regulatory prose is not query vocabulary.
   Needed before the register can drive retrieval directly.
3. **Filter cost.** Is 18.7% acceptable, or should observational studies be admitted
   carrying a study-design tag so downstream can weight them?
4. **Corpus shape.** Per-claim files, or one deduplicated pool plus a claim→PMID
   mapping? Undecided — the strict run found 199 unique PMIDs across 200 records,
   so overlap was near-zero at this scale and the question never became urgent.

## Picking it back up

`notebooks/01_pubmed_retrieval.ipynb` runs top-to-bottom against live PubMed; all
parameters are in section 1. Requires `NCBI_API_KEY` and `NCBI_EMAIL` in `.env`
(see `.env.example`); it works without a key at the slower throttle.

Note that the DSLD work that followed established a per-claim extraction shape with
a hand-labelled gold set (`datasets/pilot/gold-100.jsonl`, `docs/dsld-purpose-risk-findings.md`).
If PubMed extraction resumes, that gold-set/scoring pattern is the closer precedent
than anything in this experiment.
