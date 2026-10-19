# Purpose/risk extraction: quality check design

Status: draft, 2026-10-15; numbers updated 2026-10-16 from the finished full run; item sample stratified by faithfulness type 2026-10-17. Decided: the check measures **faithfulness** and **bucket
correctness** (is a purpose a purpose, is a risk a risk); judges are Claude Code and
Codex subagents with a human co-judge. Everything else here is proposed.

## What is checked

Qwen3.8-27B extracted `purpose` and `risk` items from 214,780 DSLD label passages
(run `full214780-r4-2610131822-0`, renderer r4): 466,630 purpose items and 355,134
risk items. Each item is meant to be a verbatim
piece of the passage. The items are not loaded into the KG directly; a second NER
extraction runs on them, so an item must be meaningful on its own, without the
passage around it.

The rubric is the extraction prompt itself
(`engine/repositories/prompts/purpose_risk/prompt.md`):

- **purpose**: an effect on the body or health the label claims, or a stated use.
- **risk**: a caution about taking the product (who should not, when to stop,
  side effects, interactions, dose or duration limits).
- **neither** (rule 4): packaging, storage, keep out of reach of children,
  certifications, disclaimers, allergen statements, ingredients, taste, quality,
  marketing, directions without a caution, product types.

## Coverage against iDISK

Of the 214,780 labels, those that iDISK 2.0 already knows (their DSLD ID appears in a
`DSLD_ID` cell of `datasets/iDISK2.0/Data/Entity/DSP.csv`, which can hold several IDs
separated by pipes) and those it does not:

| | labels | share | any purpose/risk | with purpose | with risk | purpose items | risk items |
|---|---|---|---|---|---|---|---|
| in iDISK | 156,943 | 73.1% | 148,731 | 116,243 | 128,361 | 366,206 | 263,948 |
| new (not in iDISK) | 57,837 | 26.9% | 48,194 | 33,822 | 43,146 | 100,424 | 91,186 |

iDISK references 156,957 DSLD IDs; 14 of them are missing from the DSLD dump.

The extraction adds information even for labels iDISK already has. iDISK fills
`Purpose`/`Risk` only from LNHPD: of the 92,650 iDISK DSP rows that carry a DSLD ID,
only 208 have a `Purpose` and 188 a `Risk`. So nearly all extracted purpose and risk
items fill fields that iDISK leaves empty, and the 57,837 new labels also add the
products themselves.

## False positives and false negatives

Errors are counted at two levels, because they answer different questions.

### Item level: is each extracted item right?

| outcome | meaning |
|---|---|
| **TP** | the item is a real claim, in the right bucket, and faithful to the label |
| **FP** | the item is wrong, one of: |
| ↳ *not a claim* | rule-4 text extracted as purpose or risk (e.g. an allergen statement as a risk) |
| ↳ *wrong bucket* | a purpose filed as a risk, or the reverse |
| ↳ *unfaithful* | the item is not supported by the passage (invented), or the cut changed its meaning (dropped a negation, condition or scope) |
| **FN** | a claim on the label that was not extracted |
| **TN** | not defined: "every piece of text correctly not extracted" cannot be counted |

Because TN is undefined, an item-level **false positive rate** does not exist. The
item-level measure is **precision** (TP / (TP + FP)) and its complement, the
**false discovery rate**. Item-level FN would need every claim on every sampled label
annotated; that is out of scope.

### Label level: did the label get the right *kind* of claim at all?

For each kind K (purpose, and separately risk), a label is a binary case:

|  | label states K | label states no K |
|---|---|---|
| **model extracted ≥1 K item** | TP | **FP**: everything extracted as K is wrong |
| **model extracted no K item** | **FN**: a K claim was missed | TN |

Here TN is well defined, so the full confusion matrix is available: FPR, FNR,
precision, recall. The label-level FN is the miss that matters most downstream: a
label with no extracted claim gives NER nothing to work on.

## The four groups

The model's output puts every label in one of four groups by which kinds it
extracted. Over the full run (all 214,780 labels `ok`):

| group | labels | share | median passage words |
|---|---|---|---|
| both | 124,647 | 58.0% | 156 |
| purpose only | 25,418 | 11.8% | 104 |
| risk only | 46,860 | 21.8% | 85 |
| neither | 17,855 | 8.3% | 26 |

Each group is the model's *prediction*, so it decides which confusion-matrix cells
its labels can land in:

| group | purpose cell | risk cell |
|---|---|---|
| both | TP or FP | TP or FP |
| purpose only | TP or FP | TN or **FN** |
| risk only | TN or **FN** | TP or FP |
| neither | TN or **FN** | TN or **FN** |

So the groups are the sampling strata: FN can only be found in the groups where the
model extracted nothing of that kind, and FP only where it extracted something.
Sampling the same number of labels per group (proposed: 150) covers the small groups
well; overall rates are then reweighted by each group's share of the catalog:

    rate_overall = Σ_g  share_g × rate_g

## The item sample: four faithfulness types

The deterministic check (below) puts every item in one of four faithfulness types. A
plain random item sample would be almost all verbatim (97.5% of purposes), leaving a
handful of stitched items and likely no spread or invented ones, so their error rates
could not be measured. The item sample is therefore stratified by type, separately for
purpose and risk, the same way the label sample is stratified by group:

| type | meaning | purpose items | risk items |
|---|---|---|---|
| verbatim | the item occurs in the passage character for character | ~455,000 | ~353,600 |
| stitched | not verbatim; every word from one passage segment, some words in between left out | ~10,400 | ~1,070 |
| spread over segments | not verbatim; every word in the passage, from more than one segment | ~510 | ~180 |
| invented words | at least one word not in the passage | ~650 | ~320 |

Counts are the full run's rates times its item totals. Proposed: 100 items per type and
kind, 800 in all. Each type's error rate is reported on its own; overall precision is
reweighted by each type's share of the kind, as for the label groups:

    precision_overall = Σ_t  share_t × precision_t

What each stratum answers: verbatim, whether the 97.5% / 99.6% can be used as they are;
stitched, whether leaving words out usually keeps the label's meaning or distorts it;
spread and invented, whether these items are worth keeping at all, or should simply be
dropped (together about 0.25% / 0.14% of items).

## Metrics

### Deterministic, over every item (no judge)

| metric | how | full run (purpose / risk) |
|---|---|---|
| verbatim rate | item is a substring of the passage | 97.52% / 99.56% |
| stitched rate | not verbatim, all words in one passage segment | 2.23% / 0.30% |
| cross-segment rate | not verbatim, all words in the passage but spread over several segments | 0.11% / 0.05% |
| invented rate | not verbatim, contains words not in the passage | 0.14% / 0.09% |
| source of item | DSLD path of the segment holding the item (`statements`, `ingredientRows`, `fullName`) | `fullName` 0.84% / 0% |
| rule-4 lexicon hits | keyword patterns for storage/seal, allergen, FDA disclaimer, "free from" | 0.05% / 1.7% (risk: storage/seal 1.2%, allergen 0.5%) |
| purpose/risk collision | same text under both kinds in one label | 4 labels |
| run health | failed labels, retries | 0 failed; 1 label took 3 attempts, all others 1 |

The rule-4 storage pattern matches storage directives only ("store in/at", "cool dry
place", "seal is broken", "tamper"); a bare "store"/"storage" also hits real purposes
such as "muscle glycogen stores", so these rates are not comparable with the 2026-10-15
snapshot's.

Cut-context flags (a negation just before the item, a qualifier such as "if" or "for"
just after it) are not reported as rates; they only select items for the judge.

BLEU, ROUGE, BERTScore and perplexity are not used: the first three score overlap with
a reference text, which is trivially perfect for verbatim items and does not say
whether a claim is the right kind; perplexity measures fluency, not correctness.

### Judged, on samples

| metric | level | from |
|---|---|---|
| precision, false discovery rate | item, per kind and per faithfulness type | item verdicts on the item sample stratified by faithfulness type, reweighted |
| error-type breakdown (not a claim / wrong bucket / unfaithful) | item, per faithfulness type | same |
| meaning-kept rate | item, per faithfulness type | faithfulness verdict on the same sample |
| FPR, FNR, precision, recall | label, per kind | label verdicts on the stratified label sample, reweighted |
| agreement with gold1000 | label, per kind | free baseline: label presence in the existing Claude-made reference vs the model |

Every rate is reported with a 95% confidence interval (Wilson). The ~300 items picked
by the deterministic flags are reported as their own group and never mixed into the
overall rates, since they were chosen for being suspicious.

### Judge reliability

| metric | between |
|---|---|
| raw agreement, Cohen's κ | Claude judge vs Codex judge |
| accuracy, κ | each judge vs the human, on the items the human reviewed |
| agreement-was-right rate | how often the two judges agreeing matches the human, on the audited slice |

## How the judges and the human annotator work together

```
             sampled items / labels
                      │
          ┌───────────┴───────────┐
   Claude Code judge        Codex judge         independent; neither sees the other
          └───────────┬───────────┘
                compare verdicts
          ┌───────────┴────────────┐
       disagree                  agree
          │                ┌───────┴────────┐
          │           random 10%         remaining 90%
          ▼                ▼                 ▼
     human adjudicates  human audits     final = judges' verdict
          └───────┬────────┘
            final = human's verdict
```

1. **Rubric pilot.** On ~50 items, both judges and the human label independently.
   Disagreements here are rubric problems, not judge errors: the rubric is clarified
   (and ambiguous cases logged as candidate prompt fixes) before the full run.
2. **Independent judging.** Each judge gets a self-contained prompt with the rubric,
   the passage and the item; it returns a verdict and a short reason. For a "yes, this
   label states a purpose/risk", it must quote the passage; the quote is checked
   deterministically against the passage, so a judge cannot claim a miss it cannot
   show.
3. **Routing.** Items where the judges agree are accepted; items where they disagree
   go to the human.
4. **Human review.** The human labels the disagreements and a random 10% of the
   agreements. The human labels first and sees the judges' verdicts only afterwards,
   so the human's label stays an independent check.
5. **Trust check.** The audited 10% tells how often agreeing judges are right. If that
   is high, the judge-only verdicts stand; if not, the audit is widened.
6. **Record keeping.** Each verdict stores the judge (model and version, date), the
   prompt version and the reason, so a batch can be rerun or compared.

Because the judges run as subscription subagents rather than API calls, there is no
temperature or seed control and the model version can change underneath; the
per-verdict record in step 6 is what keeps results comparable.
