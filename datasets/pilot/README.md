# Purpose/risk reference sets

`gold-1000.jsonl` is the completed **1,000-label** reference set: the original
100 rows followed by the remaining 900 rows of the seeded DSLD-only pilot.
There is one row per unique DSLD ID, in ascending ID order. The original
`gold-100.jsonl` and its manifest are unchanged.

Each row contains purpose/risk claims, exact evidence quotations, a `grounded`
flag on each claim, field confidence, and annotator/timestamp provenance.
Empty lists mean the annotation found no stated purpose or risk; labels were
not skipped or replaced with placeholder rows to reach the target count.

The bundle includes:

- `gold-1000.sources.jsonl`: all 1,000 original rendered passages, so evidence
  can be checked offline without Claude's temporary files or the bulk dump.
- `gold-1000.manifest.json`: checksums, counts, sampling provenance, annotator
  cohorts, the complete targeted review edit log, and the header audit.
- `gold-1000.instructions.md`: the frozen instructions used for the expansion.
  The original 100 predate its packaging/storage amendment.
- `validate_gold1000.py`: a standard-library check of unique IDs, source
  coverage/order, schema, confidence, verbatim evidence, checksums, preserved
  original rows, provenance, and manifest statistics.

Run the integrity check from the repository root:

```sh
python3 datasets/pilot/validate_gold1000.py
```

The 1,000 labels are the full original random pilot (seed `20260908`), not a
fresh draw. The original 100 were its lowest-ID prefix. Finishing the pilot
removes that prefix restriction, while retaining the pilot's DSLD-only
sampling frame and its held-back LNHPD/DSLD validation products.

Annotation provenance: 100 original Claude rows, 810 saved Claude expansion
rows, 75 recovered Claude draft rows, and 15 new Codex rows. Codex also
performed targeted consistency corrections on 26 expansion rows; each edit
is recorded in the manifest. `model` records the original annotator, and the
manifest records the later reviewer. Recovered draft timestamps come from
the draft files' modification times.

There are **1,979 purpose claims and 2,302 risk claims**. All **4,281** evidence
spans occur exactly in their source label's statement text; none requires a
metadata header or bracketed statement-type prefix. This supports examining
header removal, but does not establish that model predictions would be
unchanged without that context.

This is a model-annotated reference, not independently human-adjudicated
ground truth. The original 100 and expansion differ in prompt version and
claim granularity; the final 15 have a different annotator. Verbatim grounding
checks quotations, not semantic completeness. The manifest makes these
differences explicit. Existing pipeline defaults still use `gold-100.jsonl`;
select `gold-1000.jsonl` explicitly when evaluating against the expanded set.
