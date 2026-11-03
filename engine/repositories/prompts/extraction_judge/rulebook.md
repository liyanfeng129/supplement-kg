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
