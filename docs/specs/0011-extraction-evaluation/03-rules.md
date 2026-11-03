# 0011.03 Evaluation rules: what the LLM judges and the human both decide, and how

- **Status:** design in progress (grilling 2026-10-26); Q1–Q16 agreed, Q17–Q21 proposed
- **Date:** 2026-10-26
- **Part of:** [0011 overview](00-overview.md). Replaces the overview's
  [Verdicts](00-overview.md#verdicts) and the label case of its
  [Judge prompts (draft)](00-overview.md#judge-prompts-draft); the overview is to be updated to
  point here.
- **Scope (proposed):** new `engine/repositories/prompts/extraction_judge/rulebook.md`, the one
  source both the judge's system prompt and the human's guide are rendered from. The judge
  service, the human sheet and the verdict store are specs of their own.

Decisions are tagged as in the overview: **[agreed]**, **[proposed]**, **[open]**. Question
numbers (Q1–Q21) are the grilling session's.

## Why one rulebook

Two Claude/Codex extractions that agree are not thereby right, and two that disagree are not
thereby wrong: agreement measures consistency, not correctness. Correctness is a judgement of
Qwen's extraction against the passage, and there is no expert ground truth for it: the human
is a careful reviewer, not a domain expert, so the human's verdicts are the reference each judge
is measured against, not the truth. So the LLM judges and the human judge **the same thing by
the same rules, in the same format**: then a judge–human disagreement is two applications of
one rule to one passage, not two readings of "claim", and blind adjudication decides which side
misapplied it, or that the rule is unclear.

## What is judged **[agreed, Q2, Q7]**

The unit is **one label**: the passage plus all of Qwen's purpose and risk claims for it. The
judge gives a verdict on every claim, then lists what Qwen missed.

A judge sees the **r4 passage** Qwen saw (lowercased, punctuation stripped), never the original
label; Qwen's claims, not said to be Qwen's; never the stratum (spec
[02](02-sample.md#why-stratifying-on-the-runs-own-output-is-safe), rule 3). The human sees the
same.

## The rules **[agreed, Q1]**

The rulebook copies Qwen's prompt (`engine/repositories/prompts/purpose_risk/prompt.md`) in its own
words, leaving out its output instructions and worked example, numbered so a verdict can cite it, with examples under the rules they illustrate,
and adds only **decision rules** on how to judge, never on what a purpose or risk is: Qwen is
judged on the task it was given. A case where the prompt itself is unclear, including any
widening of purpose or risk, is logged as a candidate prompt fix for a later Qwen run, not
fixed here **[agreed 2026-10-28]**.

### Rule ids

| id | prompt text (abridged; the rulebook holds the prompt's own words) | the rulebook's examples |
|---|---|---|
| `def.purpose` | an effect on the body or health the label claims the product has, or a use it states the product is for | |
| `def.risk` | a caution about taking the product: who should not, when to stop or see a doctor, side effects, interactions, limits on dose or duration | "not for children under 18", "for adults only", "consult a physician if pregnant, nursing or taking medication", "do not exceed recommended dose", "take no more than 2 daily", "never consume more than in a hour period" (a limit, even with its number lost from the text) |
| `R1` | use only what the label states; never infer | |
| `R2` | copy exactly, one continuous piece; the shortest piece that states the whole claim, without lead-ins or trailing asterisks | |
| `R3` | one claim per line; split where the label states several, unless the split needs words the label does not repeat | |
| `R4a` | text equally true of any product is not a claim: packaging, seals, storage, keep out of reach of children, origin, contact details, certifications, regulatory disclaimers | "keep out of reach of children", "do not use if seal is broken", "store in a cool dry place", "these statements have not been evaluated by the FDA ..." |
| `R4b` | text describing the product, not its effect on the person, is not a claim: ingredients, allergens, taste, quality, how it is made, marketing adjectives, paired products, directions unless they carry a caution | "contains soy", "made in a facility that also processes nuts", "minerals are chelated bonded to promote mineral absorption", "prolonged release ... promoting optimal assimilation" (its form making its own ingredients easier to absorb), "take as desired up to four times per day" (a direction, with no caution) |
| `R4c` | a purpose must say what the product does for the body or health; a slogan, product type or feature is not one | "dietary supplement", "protein supplement", "for use as a multivitamin", "glucose transport and lean tissue support complex" (the name of a blend) |
| `R5` | no purpose or no risk stated: no line for it | |

The examples sit under their rule in the rulebook, marked `e.g.`, apart from the prompt's
own text.

### Decision rules **[agreed, Q11, Q13; cut to D0, agreed 2026-10-28; D1 added, agreed 2026-11-02]**

| id | case, e.g. | decision |
|---|---|---|
| `D0` | a negation, condition, scope (who, when) or amount dropped from the claim | `meaning_changed` (R2 asks for the whole claim) |
| `D1` | a claim reworded with the label's meaning kept | `faithful` `yes` (Q12: the copy rule is measured by the profile's types) |

D0 and D1 say how to fill `faithful`, not what a claim is, so they stay; D1 is not the old D1 below. Everything else that stood
here is gone (2026-10-28):

- the old D5–D11 restated `def.risk`, `R4a`, `R4b` and `R4c`, and became their examples;
- the old D1–D4 and the purpose line (Q16) widened purpose beyond the prompt's definition,
  so a judge would hold Qwen to rules Qwen never saw. They are the first candidate prompt
  fixes below.

### Candidate prompt fixes

Cases the extraction prompt leaves unclear, for a later Qwen run's prompt; the judges and the
human judge these by the prompt's definitions as they stand, and an unclear case is routed
by disagreement or `low` confidence as any other.

| case, e.g. | proposed for the prompt |
|---|---|
| vague wellness or lifestyle wording: "supports longevity", "support your everyday health", "support your inner beauty", "nurture your health and vitality", "replace your daily multi" | purpose (vague, still a purpose or a use) |
| appearance: "supple skin", "lustrous hair", "strong nails" | purpose |
| who it is made for: "formulated for those with chemical sensitivities" | purpose (a use) |
| the product lacks a side effect: "without the flushing effect associated with niacin" | open |
| directions that name who takes it: "adults take one capsule daily" | open |
| the product's form makes its own ingredients easier to absorb: "minerals are chelated bonded to promote mineral absorption", "prolonged release ... promoting optimal assimilation" | neither (R4b; a rulebook example since the pilot, not yet in the prompt) |
| a dose in the directions: "take as desired up to four times per day" a direction, "never consume more than in a hour period" a limit | as worded: a limit only with "never", "do not exceed", "no more than" (rulebook examples since the pilot, not yet in the prompt) |
| a general health statement that does not name the product: "replacing ... foods with wholesome paleo foods promotes better weight management" | open |

From the human's reveal decisions on the pilot (2026-11-03). Drafted from those decisions for a
later prompt, not yet reviewed as prompt changes; the rulebook stays frozen.

| case, e.g. | the human decided | proposed for the prompt |
|---|---|---|
| a bare health topic: "heart health", "cardiovascular health", "muscle function", "bone support", "whole body wellness energy support" | purpose | purpose |
| an effect tied to one ingredient: "conjugated linoleic acid cla to preserve lean mass", "medium chain triglycerides mcts for fast energy", "malic acid is essential in the formation of atp" | purpose | purpose (the effect is stated; R4b's "ingredients" means the list of them) |
| an ingredient that helps absorb other things, or absorption in general: "bioperine enhances absorption of whole food and supplements", "maximum absorption and retention", "supports maximum absorption without stomach irritation" | neither | neither (widens R4b's example, which covers a form and its own ingredients) |
| a caution given as a direction: "dilute with a carrier oil to minimize skin sensitivity" risk; but "do not use as a sole source of nutrition", "as directed by your healthcare provider", "do not drink directly from the bottle as it will cause contamination" not a claim | mixed | open: where a direction carries a caution |
| a name or slogan that states an effect: "fat burning formula", "performance recovery carb fuel" purpose; but "weight management program" neither | mixed | open: R4c against def.purpose's "a use it states" |
| comfort or the absence of a problem: "you don t have to worry about cramping or dehydration" purpose; but "gentle on the stomach" neither | mixed | open, with the side-effect row above |
| general replenishment and beauty copy: "replenish those nutritional reserves which the body loses through time" neither; "cosmetic coverups cannot ... support your inner beauty" not a claim; but "for optimal health" purpose | mixed | open, with the vague-wording row above |

Two decisions go against the frozen rulebook as written, both kept against both judges. They
are the human's to keep, and if the human set repeats them they are a rulebook question, not a
judge error: "never consume more than in a hour period" as `neither` (def.risk gives this very
line as a limit on dose), and "gently detoxifies" as `invented` from the label's "gentle
detoxifier" (D1: a rewording with the meaning kept is `yes`).

## The verdict

### Per claim **[agreed, Q3, Q5, Q6, Q10, Q12; `merged` dropped and `both` added, agreed 2026-10-28]**

Two questions per claim, each measured against something else: does the label say this
(against the label), and what is it (against the purpose and risk definitions). Extra text, a
lead-in, and two claims of one kind on one line are left to entity extraction downstream,
which splits and trims the claim text; a dropped negation, condition, scope or amount, or a
line mixing a purpose and a risk, is not, so those stay errors.

| field | values | notes |
|---|---|---|
| `claim_id` | `claim:<dsld_id>:<kind>:<i>` | as in spec 02 |
| `faithful` | `yes` · `invented` · `meaning_changed` | judged on meaning only (Q12): a stitched or spread claim whose meaning is kept is `yes`; the copy rule is already measured by the profile's types. A dropped negation, condition, scope or amount is `meaning_changed` (D0) |
| `actual_kind` | `purpose` · `risk` · `neither` · `both` | the claim as the label means it, by the rules above; empty when `invented`, as there is nothing on the label to classify. `both`: the line holds a purpose and a risk, e.g. `supports joint health do not use if pregnant`. A line holding several purposes, e.g. `eases tension and settles the stomach`, is `purpose`; several risks, `risk` |
| `note` | `rule` (a rule id), `evidence` (an exact quote from the passage, checked automatically), `reason` (one sentence) | required unless the claim is `faithful yes` and of Qwen's kind |
| `confidence` | judge: `high` · `medium` · `low`; human: `unsure` yes/no | |

**Scored as two measures, not one outcome [agreed 2026-10-28]:** the two fields measure
different things, so neither is ranked over the other.

| measure | computed as | tells |
|---|---|---|
| faithfulness | share of claims with `faithful = yes`; errors split `invented` / `meaning_changed` | whether Qwen copies the label honestly |
| kind accuracy | share of claims whose `actual_kind` is Qwen's kind; errors split wrong kind / `neither` / `both` | whether Qwen tells purpose from risk |
| usable | `faithful = yes` and `actual_kind` is Qwen's kind | the share of claims fit for the KG as they are |
| recall | usable claims / (usable + missed) | how much Qwen leaves out |

An `invented` claim has no kind, so it is left out of kind accuracy. These replace the
derived outcome and its priority (unfaithful > not a claim > wrong kind), agreed before.

Only `faithful` and `actual_kind` are compared between judges and with the human. The `note`
is for audit, never scored: a written reason is a justification, not proof of how the verdict
was reached.

There is no other boundary verdict: a lead-in or trailing text that keeps the meaning is
`faithful yes`. A claim extracted twice is not judged twice: an exact repeat is
caught automatically, and a repeat in other words is left to entity deduplication downstream.

### Per label: missed claims **[agreed, Q9]**

After the claims, every claim Qwen missed, split as finely as R3 asks Qwen to:

| field | values |
|---|---|
| `quote` | an exact piece of the passage; checked automatically |
| `kind` | `purpose` · `risk` |
| `rule` | the definition it falls under (`def.purpose` or `def.risk`) |
| `confidence` | as above |

A quote that overlaps one of Qwen's claims is not missed: a second claim inside a Qwen line
is part of that claim (a second kind there makes it `both`), and the check rejects the quote. Whether the label states any purpose or any risk is derived (Qwen's
correct claims ∪ missed claims), not asked.

### Example

Label 62375 (Niacin 500 mg Flush Free Formula; in the pilot, not in the human set). The r4
passage:

```
niacin flush free formula
warning not intended for pregnant or nursing women
inositol hexanicotinate
no added sugar salt yeast preservatives artificial flavors or colors
niacin vitamin b3 plays a key role in converting food into energy and helps maintain the healthy function of the nervous system the digestive system and skin
niacin as inositol hexanicotinate is a good source of vitamin b3 without the flushing effect associated with niacin
helps convert food into energy
vitamin health
gluten free
directions as adults take one capsule daily with any meal or as directed by your healthcare provider
niacin form as inositol hexaniacinate
inositol form as inositol hexaniacinate
```

Qwen's claims, shown without saying they are Qwen's:

```
1 P: plays a key role in converting food into energy
2 P: helps maintain the healthy function of the nervous system
3 P: helps maintain the healthy function of the digestive system
4 P: helps maintain the healthy function of skin
5 P: helps convert food into energy
6 R: not intended for pregnant or nursing women
```

A plausible verdict:

| # | faithful | actual_kind | comment |
|---|---|---|---|
| 1 | yes | purpose | |
| 2 | yes | purpose | |
| 3 | yes | purpose | stitched from the shared "helps maintain the healthy function of"; meaning kept, so `yes` (Q12) |
| 4 | yes | purpose | stitched, as 3 |
| 5 | yes | purpose | same meaning as 1, stated again on its own line; not an exact repeat, so left to entity deduplication |
| 6 | yes | risk | |

Missed, both open:

- `without the flushing effect associated with niacin`: purpose, or neither as a description
  of the product (R4b)?
- `adults take one capsule daily`: risk (who may take it, a dose; `def.risk`), or neither as a
  direction (R4b)?

Six usable claims (faithful, and of Qwen's kind), zero to two misses. All six claims are easy and the judges should
agree. The two candidate misses are where they may disagree or give `low` confidence, so they
are routed to the human. If the judges and the
human all hesitate on the same case, the prompt is unclear there: both are logged as
[candidate prompt fixes](#candidate-prompt-fixes).

## Who judges, and how verdicts become final

- **Two LLM judges, independent [proposed, Q18]:** Claude and Codex each judge every label,
  blind to each other. Each verdict records the judge, its model id and version, the rulebook
  hash, the date and the attempt. **Pinned [agreed 2026-11-03]:** `claude-sonnet-5-5` and
  `gpt-6.1-terra`, both at medium effort (`configs/*.toml`), the models behind the pilot's
  last verdicts, for the large set too.
- **The human, same rules, same format [agreed, Q7; the answer key as a verdict proposed,
  Q17]:** on the 20 human pilot labels and the 150-label human set, blind, the human gives the
  same verdict (claims + missed). This replaces writing a fresh extraction.
- **Routing to the human [agreed for the pilot, Q14]:** every claim on which the two judges
  disagree on `faithful` or `actual_kind`, every missed claim only one judge lists or the two
  give different kinds, and anything either gives `low` confidence.
- **No routing on the large set [agreed 2026-11-03]:** the human reviews the 150-label human
  set only; judge–human disagreement is measured there. Routing all would take about 670 more
  labels, about 16 hours at the pilot's 1.4 minutes per label (14 of the 30 pilot labels outside
  the human part were routed).
- **Unresolved [agreed 2026-11-03]:** outside the human set, whatever the routing rule above
  catches gets no human verdict. It is reported as `unresolved`, and the metrics give the range
  from best to worst case over it, rather than picking a judge or dropping the case.
- **Blind adjudication [agreed, Q15]:** on a routed case the human gives their own verdict
  first; only then are the judges' verdicts and reasons shown, to log a rulebook gap.
- **Final verdict [agreed 2026-11-03]:** on a label the human reviewed, the human's decision
  where there is one, otherwise their blind verdict. Elsewhere, the judges' verdict where they
  agree, otherwise `unresolved`. Agreement is always measured on the blind verdicts, never the
  final ones.
- **Confidence is checked before it routes [agreed, Q6]:** on the pilot, are low-confidence
  verdicts the ones that disagree with the human more often? If not, confidence is dropped from
  routing. **Result (2026-11-03): too little data to drop it, so it stays until the human set
  [agreed 2026-11-03].** Claims against the blind verdicts, on the 36 reviewed labels: Claude
  differs on 2 of 3 `low`, 3 of 29 `medium`, 6 of 176 `high`; Codex on 7 of 10 `medium`, 12 of
  198 `high`, and it gave no `low`. Missed claims cannot be tested against the blind verdicts:
  the human listed only 2, neither also listed by a judge. Against the later decisions, which are
  not blind, Claude's quotes became claims on 7 of 20 `low` and 7 of 9 `medium`; Codex's on 7 of
  15 `medium` and 17 of 22 `high`.

## The rulebook file **[agreed, Q8; layout proposed, Q21]**

One markdown file, `engine/repositories/prompts/extraction_judge/rulebook.md`: the prompt's
definitions and rules, without its output instructions and worked example, with their ids and examples, D0 and D1, and how to fill each field. The judge's system prompt and the human's guide are both rendered from it, so
they cannot drift. Every verdict records its hash.

During the pilot the rulebook may change (examples or judging rules from disagreements; a
definitional gap goes to the candidate prompt fixes instead); each change
re-judges all 50 pilot labels. It is frozen before the large set.

**Frozen at `aa9b5c22f611daa3` [agreed 2026-11-03]:** the pilot is closed, with no further round
of changes. What the reveal showed went to the candidate prompt fixes, not the rulebook.

## How the judge is called **[proposed, Q19]**

1 label per call for both judges (2026-10-28), so they are compared on the same input. Codex
at 5 moved text between the labels of a call; with its added context cut to ~3k tokens per
call, batching saves little. Claude's earlier batch-5 pilot verdicts are kept beside the run's
judge file (`judge_pilot_claude_sonnet-5-5_batch5.jsonl`) for comparison. The answer is JSON,
fields in the order note (rule, evidence, reason) → verdict fields → confidence; the prompt's
checks (ids present once, enums closed, quotes in the passage,
missed quotes not overlapping a Qwen claim) run in the controller and a failure is re-asked
through its retry, as for the purpose/risk prompt.

### Example: what one call sends

The whole prompt of one call, exactly as sent, for pilot label 267401 (outside the human 20).
The system prompt is the same for every call: `rulebook.md` (rulebook `aa9b5c22f611daa3`)
then notebook 16's `ANSWER_FORMAT`, prompt `62191c27275e6965`, 7,923 chars. If
either changes, this copy is stale: rebuild it from notebook 16.

**System prompt:**

~~~~text
# Rulebook: judging a purpose/risk extraction

You are given the text of one supplement label and the claims extracted from it, each a line
`P: <text>` (a purpose) or `R: <text>` (a risk). Judge every claim, then list the claims the
extraction missed. Judge the extraction on the task it was given: the definitions and rules
below are the extraction prompt's own, with ids so a verdict can cite them, leaving out its
output instructions and its worked example.
The examples under a rule (`e.g.`) are the rulebook's, not the prompt's.

## Definitions

- def.purpose: an effect on the body or health that the label claims the product has for the
  person taking it, or a use it states the product is for.
- def.risk: a caution about taking the product -- who should not take it, when to stop or see
  a doctor, side effects, interactions, limits on dose or duration.
  - e.g. "not for children under 18", "for adults only" (who should not take it); "consult a
    physician if pregnant, nursing or taking medication" (see a doctor); "do not exceed
    recommended dose", "take no more than 2 daily", "never consume more than in a hour period"
    (a limit on dose, even with its number lost from the text).

## Rules

- R1: Use only what the label states. Never infer a claim from the ingredients or from your
  own knowledge. A missing claim is better than an inferred one.
- R2: Copy each claim from the label exactly, character for character, as one continuous
  piece of the text. Take the shortest piece that still states the whole claim, leaving out
  lead-ins such as "WARNING:" or "This product is a ...", and trailing asterisks.
- R3: Each line holds one purpose or one risk. Where one piece of label text states several,
  split it into one line each. If a split would need words the label does not repeat, keep
  the whole piece on one line: copying exactly comes first.
- R4a: Text that would be equally true of any product is not a claim: packaging, seals,
  storage, keeping out of reach of children, origin, contact details, certifications and
  regulatory disclaimers.
  - e.g. "keep out of reach of children", "do not use if seal is broken", "store in a cool
    dry place", "these statements have not been evaluated by the FDA ...".
- R4b: Neither is text that describes the product rather than its effect on the person:
  ingredients, allergen statements, taste, quality, how it is made, marketing adjectives,
  other products it pairs with, and directions for use unless they carry a caution.
  - e.g. "contains soy", "made in a facility that also processes nuts", "minerals are chelated
    bonded to promote mineral absorption", "prolonged release ... promoting optimal
    assimilation" (its form making its own ingredients easier to absorb), "take as desired up
    to four times per day" (a direction, with no caution).
- R4c: A purpose must say what the product does for the body or health; a slogan, a product
  type or a feature is not one.
  - e.g. "dietary supplement", "protein supplement", "for use as a multivitamin" (product
    types); "glucose transport and lean tissue support complex" (the name of a blend).
- R5: If the label states no purpose or no risk, emit no line for it.

## Decision rules

How to judge, not what a purpose or risk is: that is the definitions and rules above alone.

- D0: a negation, condition, scope (who, when) or amount dropped from the claim ->
  meaning_changed (R2 asks for the whole claim).
- D1: a claim reworded with the label's meaning kept -> faithful yes. Copying is measured
  elsewhere.

## Per claim

- faithful: does the label say this?
  - yes: it does, with its meaning kept. A claim stitched from pieces of the label, or with a
    lead-in or extra text left on, is yes when its meaning is the label's.
  - invented: the label does not say it (R1).
  - meaning_changed: the label says something like it, but the claim changes what it means,
    e.g. a dropped "not", condition, who or amount (D0).
- actual_kind: what the claim is, by the definitions and rules above: purpose, risk,
  neither, or both. Left empty when the claim is invented.
  - both: the line holds a purpose and a risk, e.g. "supports joint health do not use if
    pregnant".
  - A line holding several purposes, e.g. "eases tension and settles the stomach", is a
    purpose; several risks, a risk.

## Missed claims

After the claims: every purpose or risk the label states that no extracted claim holds,
split as finely as R3 asks, each an exact piece of the label text with its kind. A piece that
overlaps an extracted claim is not missed: a second claim inside an extracted line is part
of that line.

## Example

Label text:

    warning not intended for pregnant or nursing women
    niacin vitamin b3 plays a key role in converting food into energy and helps maintain the healthy function of the nervous system the digestive system and skin
    niacin as inositol hexanicotinate is a good source of vitamin b3 without the flushing effect associated with niacin
    helps convert food into energy

Claims:

    P: helps maintain the healthy function of the digestive system
    R: not intended for pregnant or nursing women

Both are faithful yes, of the kind given; the first is stitched from the shared
"helps maintain the healthy function of" with its meaning kept. Missed include "plays a key
role in converting food into energy" (purpose) and "helps convert food into energy"
(purpose).

## Answer format

### Input

You are given one or more labels, each as:

    Label <case_id>

    Text:

        <the label text>

    Claims:

        <claim_id> P: <text>      (extracted as a purpose)
        <claim_id> R: <text>      (extracted as a risk)

A label with no claims shows `(none)` under `Claims:`; for it, list only what it missed. Judge
each label on its own, from its own text only.

### Output

One JSON object and nothing else, no code fence. One entry per label, in the order given,
with its `case_id` exactly as given. In each, one object per claim in the order given, and one
per missed claim (`[]` when there is none), with the fields in this order:

    {"labels": [
      {"case_id": "label:62375",
       "claims": [
         {"claim_id": ..., "rule": ..., "evidence": ..., "reason": ...,
          "faithful": ..., "actual_kind": ..., "confidence": ...}
       ],
       "missed": [
         {"quote": ..., "kind": ..., "rule": ..., "confidence": ...}
       ]}
    ]}

### Claim fields

- `claim_id`: as given.
- `rule`: the id of the rule the verdict rests on, e.g. `"R4a"`, `"D0"`, `"def.risk"`.
- `evidence`: a quote of the label text the verdict rests on; `""` when there is nothing on
  the label to quote.
- `reason`: one sentence.
- `faithful`: `"yes"`, `"invented"` or `"meaning_changed"`.
- `actual_kind`: `"purpose"`, `"risk"`, `"neither"` or `"both"`; `null` when `faithful` is
  `"invented"`.
- `confidence`: `"high"`, `"medium"` or `"low"`.

`rule` and `reason` may be `""` only for a correct claim: `faithful` `"yes"` and `actual_kind`
the kind it was extracted as. Fill both for every other claim.

### Missed fields

- `quote`: a quote of the label text that states the missed claim.
- `kind`: `"purpose"` or `"risk"`.
- `rule`: `"def.purpose"` or `"def.risk"`.
- `confidence`: `"high"`, `"medium"` or `"low"`.

Split missed claims as R3 asks: one entry per claim where each is its own continuous piece of
the text; where a split would need words the label does not repeat (e.g. "consult your doctor
if you are pregnant nursing or taking medication"), quote the whole piece once.

### Quotes

`evidence` and `quote` are copied from the label's own text character for character, as one
continuous piece of one line. Never reword, shorten inside, join pieces, or quote another
label. For a claim stitched from several pieces of a line, `evidence` is that line, or the
part of it that holds all the pieces, never the claim itself.
~~~~

**User turn:**

~~~~text
Label label:267401

Text:

    lime
    for aromatic or topical use diffuse aromatically or to apply topically dilute with a carrier oil to minimize skin sensitivity
    caution possible skin sensitivity
    if pregnant or under a doctor s care consult your physician
    avoid contact with eyes inner ears and sensitive areas avoid sunlight or uv rays for up to hours after applying
    citrus aurantifolia
    essential oil supplement
    for internal use dilute one drop in fl oz of liquid
    cptg certified pure tested grade
    peel back for more information

Claims:

    claim:267401:purpose:0 P: for aromatic or topical use
    claim:267401:risk:0 R: possible skin sensitivity
    claim:267401:risk:1 R: if pregnant or under a doctor s care consult your physician
    claim:267401:risk:2 R: avoid contact with eyes inner ears and sensitive areas
    claim:267401:risk:3 R: avoid sunlight or uv rays for up to hours after applying
~~~~

The judge answers with one JSON object, `{"labels": [{"case_id": "label:267401", "claims":
[...], "missed": [...]}]}`. When a check fails, the controller sends the same prompt again
with this appended to the user turn: "Your previous answer broke these rules:", each failed
check's message as a `- ` line, then "Answer again, following the instructions exactly."

## The human sheet **[proposed, Q20]**

A small web page: the passage on one side, the claims with their verdict controls, missed
claims added by selecting text in the passage (quotes then cannot be mistyped). A spreadsheet
the repo writes and reads back is the fallback. Built in step 4: the web page, spec
[04](04-human-sheet.md).

## Not in this spec

- The judge service, its storage under `eva/`, and the metrics: the verdict format here is
  what they consume.
- The sizes of the large and human sets: spec 02's quotas, fixed after the pilot.

## Open questions

Q17 (answer key as a verdict), Q18 (two independent judges, pilot
models), Q19 (batch size, output format), Q20 (the sheet), Q21 (rulebook layout): proposed,
awaiting the user.
