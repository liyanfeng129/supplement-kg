# Near-duplicate clustering of flattened labels

- **Status:** proposed
- **Date:** 2026-10-06
- **Scope:** `engine/services/dsld_cleaning/service.py` (new `cluster` method and
  module-level entry), `engine/services/dsld_cleaning/schema.py` (new `Cluster`),
  `pyproject.toml` and `requirements-lock.txt` (new dependency `datasketch`, which
  brings `scipy`), new tests in `tests/test_dsld_cleaning.py`.

Builds on `docs/specs/0005-renderer-v4-cleaning.md`: the input is v4 flattened rows,
already normalised and without boilerplate.

Out of scope: choosing a subset from the clusters (coverage sampling with embeddings),
which comes in a follow-up spec. Also out of scope: persisting clusters to disk; the
sampling spec decides whether and how clusters are saved, since it knows what it reads
back.

## Why

Most DSLD labels are near-copies of other labels: the same product under another size,
flavour or retailer, or a reformulation with identical statements. A subset drawn at
random from the full set is mostly the same few kinds of label again. To keep as much
of the domain as possible in a small subset, collapse the copies first and sample over
what is left.

An earlier attempt sorted labels alphabetically and merged neighbours that shared 90%
of their characters. It misses copies that differ at the start (only neighbours are
compared) and copies with reordered text, and a character-percentage threshold means
something different for short and long strings. This spec replaces it with MinHash
LSH: Jaccard similarity over shingles, found across the whole set in roughly linear
time.

## What counts as a near-duplicate

Two labels are near-duplicates when the **Jaccard similarity of their shingle sets** is
at least `threshold`.

- **Text compared:** the row's `passage`, as is. Normalising and boilerplate removal
  are done by the v4 renderer (spec 0005), so clustering does none of its own; that
  keeps one definition of "the same text" for clustering and extraction.
- **Shingles:** character `k`-grams of the passage.
- **Input must be v4 or later, and one version.** A v1–v3 passage still has case,
  punctuation, quantities and boilerplate in it and would give different, worse
  clusters silently, so `cluster` refuses it. Rows rendered by two different versions
  are normalised two ways and would be compared silently too, so every row must carry
  the same `renderer_version`.
- **Short and long passages are not treated alike.** On a short passage one changed
  word is a large share of its shingles; on a long one a whole extra sentence may not
  be. So short labels split more readily and long ones merge more readily. Accepted:
  splitting short labels costs subset slots, not coverage. Revisit only if the review
  in acceptance 12 shows it doing harm.

## Clustering: star, not connected components

LSH finds similar *pairs*. Joining every connected pair into one cluster chains: A~B and
B~C put A and C together even when A and C are only 0.7 similar. So the clusters are
built as stars:

```
for row in flattened, in input order:
    if an earlier row had exactly this passage: row joins that row's cluster; next row
    candidates = LSH query over representatives only
    keep candidates whose estimated Jaccard to row ≥ threshold
    if any:  row joins the most similar one (ties: earliest representative)
    else:    row becomes a new representative and is inserted into the LSH
```

- **Every member is ≥ `threshold` to its representative** (as estimated from the
  MinHash signatures). Members are not guaranteed to be that close to each other.
- **Identical passages always share a cluster.** A row whose passage exactly matches
  an earlier row's goes straight into that row's cluster, with no shingling, signature
  or query. Most copies are exact after v4 normalising, so this skips most of the work.
  It can differ from running the row through the loop (a later copy could have found a
  newer, closer representative), but the star guarantee holds either way.
- **The representative is the first row of its cluster in input order.** The caller
  controls the order, so a caller that wants a particular row to represent its cluster
  puts it first. Picking a medoid instead was rejected for now: it costs a full
  comparison inside every cluster, and the follow-up sampling step chooses which
  members to use anyway.
- **The LSH is built at a lower threshold than the join check:**
  `max(threshold − 0.1, 0.5)`. LSH bands tuned at exactly `threshold` miss many pairs
  just above it (false negatives), and a missed pair silently becomes an extra
  representative; the verify step can drop false positives but cannot recover a pair
  that never collided. Building the LSH lower costs more candidates to verify, each a
  signature comparison. The margin is fixed, not a parameter.
- **The join check uses the signature estimate**, not exact Jaccard. At `num_perm=128`
  its standard deviation near 0.9 is about 0.027, so the boundary is soft by a few
  points; exact Jaccard would mean keeping every representative's shingle set in
  memory. A caller who wants a tighter boundary raises `num_perm`.
- **Empty passages** (a v4 label that was all boilerplate) form one cluster of their
  own and are never compared, since an empty shingle set has no meaningful Jaccard.

## Interface

```python
# engine/services/dsld_cleaning/schema.py
@dataclass(frozen=True)
class Cluster:
    cluster_id: int          # 0, 1, 2 … in order of the representative's first appearance
    representative: int      # dsld_id
    members: list[int]       # dsld_ids in input order; includes the representative first

    def to_dict(self) -> dict: ...
```

```python
# engine/services/dsld_cleaning/service.py
class DsldCleaningService:
    def cluster(self, flattened: list[dict], threshold: float = 0.9,
                shingle: int = 5, num_perm: int = 128) -> list[Cluster]:
        """Group near-duplicate labels by MinHash LSH over `passage`."""

# module level, like `clean` and `flatten`
def cluster(flattened, threshold=0.9, shingle=5, num_perm=128) -> list[Cluster]: ...
```

- Takes the output of `flatten` (or rows read back from a v4 label set): only
  `dsld_id`, `passage` and `renderer_version` are read.
- Does no I/O and changes no row. It returns clusters; the caller decides what to do
  with them, and saves a chosen subset with the existing `save`.
- Every input row is in exactly one cluster. The cluster sizes are the duplicate counts
  the sampling step needs to weight its metrics back to the full set.
- **Deterministic:** the same rows in the same order with the same parameters give the
  same clusters. MinHash uses a fixed seed.
- Raises `ValueError` on duplicate `dsld_id`s in the input, on any row whose
  `renderer_version` is missing or below 4, on rows with more than one
  `renderer_version`, on `threshold` outside (0, 1], and on `shingle` or `num_perm`
  below 1.

### Where each step lives

All in `engine/services/dsld_cleaning/service.py`, in a new `# --- clustering ---`
section after `# --- flattening ---`, the same way the cleaning steps are private
methods under `clean`. Nothing goes in `repositories/`: there is no I/O.

| Function | Does | Calls |
|---|---|---|
| `cluster(flattened, threshold, shingle, num_perm)` (public) | validates the input (ids, one renderer version ≥ 4, parameters); builds one `MinHashLSH(max(threshold − 0.1, 0.5), num_perm)`; runs the star loop over the rows (send an exact passage repeat to its twin's cluster, else query the LSH, check the estimated Jaccard against `threshold`, join or become a representative and insert); handles empty passages; returns the `Cluster`s | `_shingles`, `_signature` |
| `_shingles(text, k) -> set[str]` (static) | every character `k`-gram of the passage; a text shorter than `k` is one shingle, itself | — |
| `_signature(shingles, num_perm) -> MinHash` (static) | builds the MinHash with the fixed seed and adds every shingle in one `update_batch` call | — |

The LSH stays inline in `cluster`, not in a helper: it exists only for the length of
one call, and the star loop is the part that reads it. Pulled out, it would be a
helper that is called once and takes the loop's state as arguments.

`datasketch` is imported only in `service.py`. `Cluster` holds plain ids, so nothing
from `datasketch` leaks out of the service.

### Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `threshold` | minimum (estimated) Jaccard similarity between a member and its representative | 0.9, provisional: set from the sweep in acceptance 11 |
| `shingle` | `k` of the character `k`-grams | 5 |
| `num_perm` | MinHash signature length; higher is more accurate and slower | 128 |

`datasketch` derives the LSH bands and rows from the LSH threshold and `num_perm`; they
are not exposed.

## Dependency

`datasketch` is added to `pyproject.toml`'s dependencies and to
`requirements-lock.txt`. It is a pure-Python and numpy library (MinHash, MinHashLSH)
with no model and no GPU. A hand-written MinHash was rejected: it is the same algorithm
with more code to own and test.

## Acceptance

1. Two rows with identical passages land in the same cluster.
2. Two rows whose passages differ by one word in a long passage land in the same
   cluster.
3. **No pair is lost to the LSH.** About 100 planted pairs spread over Jaccard
   0.85–0.97: every pair whose *estimated* Jaccard (from the same signatures the
   service builds) is at or above the threshold lands in one cluster. This tests what
   the LSH margin promises without depending on whether a given pair's estimate
   happens to fall above or below 0.9.
4. Two rows whose passages share no shingles land in different clusters.
5. With A, B, C where J(A,B) and J(B,C) are above the threshold but J(A,C) is clearly
   below it, A and C are not in the same cluster.
6. The first row of each cluster in input order is its representative; `cluster_id`s
   follow the representatives' order of appearance.
7. Every input `dsld_id` appears in exactly one cluster's `members`; the sizes sum to
   the number of input rows.
8. Running twice on the same input gives identical output.
9. The input rows are unchanged after the call.
10. Rows with empty passages are all in one cluster; no other row joins it.
11. Duplicate `dsld_id`s, a row below renderer v4 or without `renderer_version`, rows
    with mixed renderer versions, and out-of-range parameters raise `ValueError`.
12. **Review on the full v4 label set** (`full214780-r4`), done in conversation, not a
    script: for `threshold` in 0.7, 0.8 and 0.9, cluster all rows and rank the
    clusters by size. Read the 30 largest and 30 more spread across the rest of the
    ranking, comparing each representative against several members drawn from across
    the cluster: are they the same product, differing only in flavour, size or
    retailer? Report examples of wrong merges and a recommended threshold; the user
    confirms the default, which is then written into this spec.
