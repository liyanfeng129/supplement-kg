# silver-1000: a second purpose/risk reference set

`silver-1000.jsonl` is a **second** 1,000-label model-annotated reference set,
drawn from the same DSLD-only sampling frame as `gold-1000.jsonl` and
**disjoint from it by construction**.

## How it differs from gold-1000

|                | gold-1000            | silver-1000              |
|----------------|----------------------|--------------------------|
| Draw seed      | `20260908`           | `20260920`               |
| Renderer       | v1 (headers, `[type]` prefixes) | v2 (free text only) |
| Annotator      | Opus (+15 Codex)     | Sonnet subagents         |
| Overlap        | --                   | **0 labels**             |

Disjointness is structural, not filtered after the fact: the 1,000 gold IDs
are removed from the 156,534-ID frame *before* sampling, so the draw cannot
collide. Excluding afterwards would make the result depend on how many
collisions happened to occur, and the draw would stop being reproducible the
moment gold changed.

## The renderer caveat, stated plainly

These passages are **renderer v2**: no `PRODUCT:`/`BRAND:` header, no
bracketed `[statement type]` prefix, all statement types retained. Gold-1000
was labelled against **v1** passages, which carried those hints.

So a metric comparing silver against gold confounds two variables at once --
annotator (Opus vs Sonnet) and renderer (v1 vs v2). Treat silver as a
v2-native set, not as a drop-in second opinion on gold.

The measured effect is real: silver recovers a purpose on 63.7% of labels and
a risk on 81.4%, against gold's 72.4% and 90.8%. An audit of the no-risk rows
found only ~0.7% contain a genuine medical caution that was missed; the rest
are correct exclusions of "KEEP OUT OF REACH OF CHILDREN", seal and storage
boilerplate, which v1's `[Precautions ...]` prefix made easier to spot.

## Post-processing applied at merge

Two mechanical passes run in `merge_silver1000.py`, both logged in full to
`silver-1000.normalisation-log.json`. `evidence` is never modified.

1. **`text` normalisation.** Annotators varied on casing. Claim text is
   lowercased with leading bullets and trailing claim asterisks stripped,
   preserving acronyms (GI, EPA, DHA, ATP, AGEs). The rule was validated by
   running it over gold-1000: **4,280 of 4,281** entries are unchanged, so it
   reproduces gold's own convention rather than imposing a new one.
2. **Fallback-risk rule.** Packaging/seal and keep-out-of-reach spans are
   risks only when the label states no other risk text. Where a genuine
   caution was also extracted, the boilerplate span is dropped. This rule
   alters **zero** gold-1000 rows.

Casing drift was not fixable by prompting -- it held at ~5% before and after
the instructions were tightened -- which is why it is corrected
deterministically at merge instead of by re-running shards. The fallback rule
*was* prompt-fixable, and later shards needed far fewer corrections.

## Integrity

Every evidence span is verified verbatim against its source passage twice:
once by the shard annotator, once at merge. The merge refuses to write if any
row is missing, misordered or ungrounded.

## Raw-field provenance audit

All 3,319 extracted entries (1,456 purpose; 1,863 risk) trace to raw
`statements[].notes`. One purpose quotation spans two statement fields; all
others match an individual field after whitespace normalization. There are no
unmatched entries. See the [field audit](../../docs/dsld-silver1000-field-audit.md)
for counts by statement type, exact-path examples, and reproduction instructions.
`silver-1000.field-trace.jsonl` contains one record per claim, and
`silver-1000.field-trace.csv` provides one row per matching raw location.

## Limits

Model-annotated, **not** human-adjudicated ground truth. Verbatim grounding
checks that a quotation appears in the label, not that the extraction is
semantically complete. Membership implies no claim about a product's safety
or efficacy.
