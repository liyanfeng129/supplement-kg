# Renderer v4: normalised passages without boilerplate

- **Status:** proposed
- **Date:** 2026-10-06
- **Scope:** `engine/services/dsld_cleaning/service.py` (two new cleaning steps, a
  change to `_clean_one`, `RENDERER_VERSION = 4`, module docstring), new
  `engine/services/dsld_cleaning/boilerplate.txt` (the checklist), new tests in
  `tests/test_dsld_cleaning.py`.

Out of scope: clustering (`docs/specs/0006-near-duplicate-clustering.md`, which takes
this spec's output as its input), and the output-side boilerplate filter
(`drop_boilerplate_risks` in `engine/pipeline/lines.py`) and the prompt rule, which stay
as they are as a backstop.

## Why

The next steps are: cluster the full dump into near-duplicates, sample over the
clusters, extract, and evaluate the extraction. Both clustering and extraction are hurt
by the same two things in today's v3 passages:

- **Boilerplate.** The FDA disclaimer, "keep out of reach of children" and storage
  lines are on most labels (the disclaimer sentence alone on 142,073 of 214,780). On a
  short label they are most of the text, so two unrelated products look alike to a
  similarity measure, and boilerplate was 33 of 38 false-positive risk claims on
  gold-100.
- **Surface variation.** `500 mg` vs `1,000 mg`, `Vitamin C.` vs `vitamin c` make copies
  of one label look different.

## What changes, and what does not

**Only raw DSLD is immutable.** Flattened label sets are versioned outputs of the
renderer. v4 is a new renderer version; label sets saved under v1–v3 (gold1000-r1,
silver1000-r2, …) stay on disk as they are and are not re-rendered. No comparability
with gold or silver is required: evaluation from here on is on the new
cluster-sample-extract path.

This reverses two statements in the module docstring ("No boilerplate filtering happens
here", and the passage being kept as the label wrote it). The docstring is rewritten
to say what v4 does and why.

**Grounding still holds.** The model reads the v4 passage and its quotes are checked
against the same v4 passage, so a faithful quote is still a substring.

**Selection is unchanged.** The same five paths are read; no statement type is dropped.
v4 changes the *text* of each segment, not which segments are read.

## Cleaning steps, in order

For each `(text, path)` found by `_name_segments`, `_statement_segments` and
`_ingredient_segments`:

1. **Normalise** (`_normalise`), every segment including `fullName`:
   1. lowercase;
   2. remove quantities (below);
   3. replace every character that is not a letter, digit or space by a space;
   4. collapse whitespace.
2. **Strip boilerplate** (`_strip_boilerplate`), every segment **except `fullName`**:
   cut each checklist phrase that occurs in the normalised text, then collapse
   whitespace again.
3. **Drop** the segment if nothing is left.

Then the existing `_deduplicate` runs over what survives. It now merges more, since
segments that differed only in case, punctuation or a number are identical after
normalising; their `paths` are merged, so provenance survives.

`CleanedLabel.name` and the row's `name` stay **raw**: they are provenance, not model
input. The `fullName` *segment* is normalised like any other.

A label whose every segment was removed keeps its row with `passage == ""` and
`segments == []`. Every raw record still has exactly one row; clustering puts these in
its empty-passage cluster, whose size is the count of boilerplate-only labels.

### Quantities

Applied to the lowercased text, before punctuation is removed (so `1,000` and `2.5`
are still one number).

- **Removed:** a number with a unit, optionally a range, with or without a space:
  `500 mg`, `500mg`, `1,000 iu`, `2.5 g`, `2-3 capsules`, `30 servings`, `10%`.
  Units: `mg mcg µg ug g kg iu ml l fl oz oz lb lbs cal kcal % cfu billion million
  capsule(s) caps tablet(s) softgel(s) vegicap(s) gummy gummies drop(s) scoop(s)
  packet(s) serving(s) teaspoon(s) tsp tablespoon(s) tbsp`.
- **Removed:** a standalone number, one not touching a letter or a hyphen on either
  side (`take 2 daily` → `take daily`).
- **Kept:** digits touching a letter or a hyphen, which are names, not amounts:
  `b12`, `b6`, `coq10`, `omega-3`, `5-htp`, `d3`.
- **Kept:** age limits. A number directly after `under`, `over`, `below`, `above`,
  `aged`, `age`, `ages` or `age of`, or directly before `year(s)`, `yr(s)` or
  `month(s)`: `not for use by persons under the age of 18`, `children under 12`,
  `ages 4 and up`. These are cautions, and the mined candidates showed that without the
  exception every age limit in the dump is lost.

### The checklist

`engine/services/dsld_cleaning/boilerplate.txt`: one phrase per line, written in
**normalised form** (lowercase, no punctuation, no quantities). Blank lines and lines
starting with `#` are ignored.

- **Matching:** a phrase matches on word boundaries only (`keep out of reach of
  children` does not match inside a longer word). Phrases are tried **longest first**,
  so `warning keep out of reach of children` is cut whole before the shorter phrase
  could leave `warning` behind.
- **Only the phrase is cut.** The rest of the segment stays. This is what keeps the
  real caution in a mixed line ("keep out of reach of children discontinue use if
  rapid heartbeat occurs" → "discontinue use if rapid heartbeat occurs").
- **Loaded once**, at import, into one compiled pattern. It is a fixed data file
  shipped with the service, not a data source, so it is not a repository.
- **Editing the checklist changes the output, so it bumps `RENDERER_VERSION`.**
- **Seeded from** `docs/boilerplate-candidates.md`: the 400 most frequent normalised
  sentences across the full dump, with the lines the user ticked becoming the file.
  Frequency alone does not decide: real cautions ("discontinue use two weeks prior to
  surgery", the iron overdose warning) and ingredient forms ("vitamin b12 form as
  cyanocobalamin") are among the most frequent lines and must stay.

## Where each step lives

All in `engine/services/dsld_cleaning/service.py`, under `# --- cleaning steps ---`.

| Function | Does | Calls |
|---|---|---|
| `_clean_one(record)` (changed) | as today, but each found `(text, path)` is normalised, boilerplate-stripped unless the path is `fullName`, and dropped if empty, before `_deduplicate` | `_normalise`, `_strip_boilerplate`, `_deduplicate` |
| `_normalise(text) -> str` (static, new) | lowercase, quantities out (with the age-limit exception), punctuation → space, collapse whitespace | — |
| `_strip_boilerplate(text) -> str` (static, new) | cuts every checklist phrase (word boundaries, longest first) and collapses whitespace | — |

`_collapse_whitespace` stays: the readers still call it on raw text, and `_normalise`
ends with the same collapse.

`RENDERER_VERSION = 4`, with its comment extended: "4 = normalised text, quantities
removed, checklist boilerplate cut."

## Acceptance

1. `Vitamin C.` and `vitamin c` normalise to the same text; so do `500 mg` and
   `1,000 mg` in otherwise identical segments.
2. `b6` and `b12`, `omega-3` and `omega-6`, `coq10` and `5-htp` survive normalising
   and stay distinct.
3. Age limits survive: `under the age of 18`, `children under 12`, `ages 4 and up`.
4. A pure-boilerplate segment ("Keep out of reach of children.") is dropped.
5. A mixed segment keeps its real caution exactly: "Keep out of reach of children.
   Discontinue use if rapid heartbeat occurs." → `discontinue use if rapid heartbeat
   occurs`.
6. Longest phrase first: with both `keep out of reach of children` and `warning keep
   out of reach of children` on the list, `warning keep out of reach of children` is
   cut whole.
7. A checklist phrase in `fullName` is not cut.
8. Two segments that become identical after normalising are one segment with both
   paths.
9. A label whose segments are all boilerplate gives one row with `passage == ""`.
10. The row's `name` is the raw `fullName`; rows carry `renderer_version == 4`.
11. On the full dump: the passage-length distribution before and after, the number of
    empty passages, and a hand-checked sample of v4 passages are recorded. This is a
    measurement for the user to run, not a unit test.
