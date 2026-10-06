# Near-duplicate clustering of flattened labels

- **Status:** proposed
- **Date:** 2026-10-06
- **Scope:** `services/dsld_cleaning/service.py` (new `cluster` method and module-level
  entry), `services/dsld_cleaning/schema.py` (new `Cluster`), `pyproject.toml` (new
  dependency `datasketch`), new tests in `tests/test_dsld_cleaning.py`.

Out of scope: choosing a subset from the clusters (coverage sampling with embeddings),
which comes in a follow-up spec. Also out of scope: persisting clusters to disk.

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

- **Text compared:** the row's `passage` (the whole label as the model sees it: name,
  statements, ingredient free text), after normalising.
- **Normalising** (for comparison only, see below): lowercase; replace every run of
  digits, with any decimal point, by `0`; replace every character that is not a letter,
  digit or space by a space; collapse whitespace. `500 mg` and `1,000 mg` then compare
  equal, as do `Vitamin C.` and `vitamin c`.
- **Shingles:** character `k`-grams of the normalised text.

**The passage itself is never changed.** Normalising builds a comparison key and is
thrown away. Grounding is a substring test against the passage, so a stored passage
that was lowercased or had its numbers replaced would fail every faithful quotation.

## Clustering: star, not connected components

LSH finds similar *pairs*. Joining every connected pair into one cluster chains: A~B and
B~C put A and C together even when A and C are only 0.7 similar. So the clusters are
built as stars:

```
for row in flattened, in input order:
    candidates = LSH query over representatives only
    keep candidates whose estimated Jaccard to row ≥ threshold
    if any:  row joins the most similar one (ties: earliest representative)
    else:    row becomes a new representative and is inserted into the LSH
```

- **Every member is ≥ `threshold` to its representative** (as estimated from the
  MinHash signatures). Members are not guaranteed to be that close to each other.
- **The representative is the first row of its cluster in input order.** The caller
  controls the order, so a caller that wants a particular row to represent its cluster
  (a gold label, say) puts it first. Picking a medoid instead was rejected for now: it
  costs a full comparison inside every cluster, and the follow-up sampling step chooses
  which members to use anyway.
- The LSH candidates are checked against the signature estimate before joining. This
  drops LSH's false positives (pairs that collided in a band but are below the
  threshold).
- **Empty passages** (nothing left after normalising) form one cluster of their own and
  are never compared, since an empty shingle set has no meaningful Jaccard.

## Interface

```python
# services/dsld_cleaning/schema.py
@dataclass(frozen=True)
class Cluster:
    cluster_id: int          # 0, 1, 2 … in order of the representative's first appearance
    representative: int      # dsld_id
    members: list[int]       # dsld_ids in input order; includes the representative first

    def to_dict(self) -> dict: ...
```

```python
# services/dsld_cleaning/service.py
class DsldCleaningService:
    def cluster(self, flattened: list[dict], threshold: float = 0.9,
                shingle: int = 5, num_perm: int = 128) -> list[Cluster]:
        """Group near-duplicate labels by MinHash LSH over `passage`."""

# module level, like `clean` and `flatten`
def cluster(flattened, threshold=0.9, shingle=5, num_perm=128) -> list[Cluster]: ...
```

- Takes the output of `flatten` (or rows read back from a label set): only `dsld_id`
  and `passage` are read.
- Does no I/O and changes no row. It returns clusters; the caller decides what to do
  with them, and saves a chosen subset with the existing `save`.
- Every input row is in exactly one cluster. The cluster sizes are the duplicate counts
  the sampling step needs to weight its metrics back to the full set.
- **Deterministic:** the same rows in the same order with the same parameters give the
  same clusters. MinHash uses a fixed seed.
- Raises `ValueError` on duplicate `dsld_id`s in the input, on `threshold` outside
  (0, 1], and on `shingle` or `num_perm` below 1.

### Where each step lives

All in `services/dsld_cleaning/service.py`, in a new `# --- clustering ---` section
after `# --- flattening ---`, the same way the cleaning steps are private methods
under `clean`. Nothing goes in `repositories/`: there is no I/O.

| Function | Does | Calls |
|---|---|---|
| `cluster(flattened, threshold, shingle, num_perm)` (public) | validates the input; builds one `MinHashLSH(threshold, num_perm)`; runs the star loop over the rows (query the LSH, check the estimated Jaccard, join or become a representative and insert); handles empty passages; returns the `Cluster`s | `_normalise`, `_shingles`, `_signature` |
| `_normalise(text) -> str` (static) | lowercase, digits → `0`, punctuation → space, collapse whitespace | — |
| `_shingles(text, k) -> set[str]` (static) | every character `k`-gram of the normalised text; a text shorter than `k` is one shingle, itself | — |
| `_signature(shingles, num_perm) -> MinHash` (static) | builds the MinHash with the fixed seed and adds every shingle in one `update_batch` call | — |

The LSH stays inline in `cluster`, not in a helper: it exists only for the length of
one call, and the star loop is the part that reads it. Pulled out, it would be a
helper that is called once and takes the loop's state as arguments.

`datasketch` is imported only in `service.py`. `Cluster` holds plain ids, so nothing
from `datasketch` leaks out of the service.

### Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `threshold` | minimum Jaccard similarity between a member and its representative | 0.9 |
| `shingle` | `k` of the character `k`-grams | 5 |
| `num_perm` | MinHash signature length; higher is more accurate and slower | 128 |

`datasketch` derives the LSH bands and rows from `threshold` and `num_perm`; they are
not exposed.

## Dependency

`datasketch` is added to `pyproject.toml`'s dependencies and to
`requirements-lock.txt`. It is a pure-Python and numpy library (MinHash, MinHashLSH)
with no model and no GPU. A hand-written MinHash was rejected: it is the same algorithm
with more code to own and test.

## Acceptance

1. Two rows with identical passages, and two whose passages differ only in case,
   punctuation, whitespace or digits, land in the same cluster.
2. Two rows whose passages share no shingles land in different clusters.
3. With A, B, C where J(A,B) and J(B,C) are above the threshold but J(A,C) is clearly
   below it, A and C are not in the same cluster.
4. The first row of each cluster in input order is its representative; `cluster_id`s
   follow the representatives' order of appearance.
5. Every input `dsld_id` appears in exactly one cluster's `members`; the sizes sum to
   the number of input rows.
6. Running twice on the same input gives identical output.
7. The input rows are unchanged after the call.
8. Rows with empty passages are all in one cluster; no other row joins it.
9. Duplicate `dsld_id`s and out-of-range parameters raise `ValueError`.
10. On the full flattened dump, the run time and the cluster-size distribution are
    measured and recorded, and a hand-checked sample of clusters looks like real copies.
    This is a measurement for the user to run, not a unit test.
