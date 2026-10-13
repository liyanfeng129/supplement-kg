# Label embedding: three servers behind one model, a service, vectors beside the label set

- **Status:** proposed
- **Date:** 2026-10-12
- **Scope:** `engine/models/embed_model.py` (several servers), `configs/embed.toml`
  (`n_servers`), `engine/repositories/dsld/labels/` (`paths.py`, `loader.py`,
  `store.py` gain embedding functions), `.gitignore` (vectors untracked), new
  `engine/services/label_embedding/service.py`, ADR 0002 v4,
  `benchmarks/embed_throughput.py` (`--parallel` becomes `--servers`, `--shard` goes:
  the model now runs several servers itself; reads the new `log_paths`), the
  `EmbedController` module docstring (its `in_flight x batch > n_parallel` rule is
  restated as "more texts queued than the model runs at once"; no code change),
  notebook 12 (calls `run("gold1000-r4")` and reads the vectors back with `load_embedding`), tests in `tests/test_embed_model.py`, new
  `tests/test_label_embedding.py`, and the labels repository's tests.

Governed by ADR 0002 (latest version) and ADR 0003. Builds on spec 0007; where this
spec says "as 0007" or "as purpose/risk", the behaviour is copied and not restated.

Out of scope: choosing the subset from the vectors (k-means, medoids), comparing
instructions, and any change to `EmbedController`'s code.

## Why

Measured on 2,000 v4 passages (`configs/embed.toml` comments, runs `embed-20261012*`):
one server with one slot does 10,675 tok/s; three single-slot servers on the same card
do ~18,200 tok/s together (1.68x), at 14.0 GB peak; a fourth adds nothing. The full
set is ~59M tokens: ~1.5 h on one server, ~55 min on three. Today only one server can
run per process, and the 2,000-text speedup was reached by splitting the ids by hand.

The vectors also have nowhere to go: notebook 12 writes an `.npz` itself because no
repository owns embeddings.

A vector is one flattened label's passage under one instruction, so it belongs to the
label set the way `annotations.jsonl` does (ADR 0003 rule 3): beside the passages it
was made from, joined to them by `dsld_id`. No new repository.

## Who owns what

| | `EmbedModel` | `EmbedController` | `LabelEmbeddingService` | `repositories/dsld/labels` |
|---|---|---|---|---|
| **Owns** | `n_servers` servers, which one answers a request, their health | unchanged (0007) | the workflow: which labels, the instruction, the embedding, resume | where a set's vectors live; reading and writing them |
| **Never knows** | what the text is | how many servers | that a server exists | what a model is |

The controller keeps calling `fits`, `submit`, `close` on one model. That the model is
three processes is the model's business (ADR 0002 §3, §8): the controller's
`in_flight x batch` (2 x 16 = 32) already exceeds the 3 texts running at once.

## The model (`models/embed_model.py`)

**One model, `n_servers` servers, one slot each.** A new setting `n_servers`
(`[model]`, default 1) replaces `n_parallel`, which is removed: one slot was the
fastest of every count measured (1 to 64, `configs/embed.toml`), so the server is the
only unit of parallelism. Each server is started as one is today, with its own port and
log, but always `-np 1` and without `--kv-unified` (one slot has nothing to share);
`n_ctx` is that slot's KV, and must stay ≥ `ubatch`. A model or card that gains from
slots brings them back as a code change.

- **`ready()`** checks the card once for `n_servers x min_vram_mib` free, then starts
  the servers one after another. Each must pass health (step 4), offload (step 6, per
  server: each start must take ≥ 90 % of the weights), well-formed vectors (7) and
  similarity (8). The end token (5) is settled on the first server and shared: the
  servers load the same file. Any failure stops every server started so far and raises
  `ModelUnavailable`. `min_vram_mib` stays per server.
- **Dispatch.** `submit` keeps one pool, now one worker per server. `embed`
  sends each request to the server with the fewest requests in flight (ties: the
  lowest index). A per-server count replaces the single `_base_url`.
- **Failure.** A request's own error is raised as today and the controller retries it,
  on whichever server is least busy then. Any server exiting, reporting CUDA / out of
  memory, or failing `/health` raises `ModelUnavailable` for the whole model: no
  server is restarted and none is dropped, as 0007.
- **`fits`** asks the first server's `/tokenize` (all share one vocabulary).
- **`close()`** stops every server. Idempotent.
- **`log_path` becomes `log_paths: list[Path]`**, one per server, in start order.

`n_servers` changes speed, not vectors: a text gets the same vector from any server
(acceptance 8). It is not in the service's `config_hash`.

## The vectors (`repositories/dsld/labels/`)

An embedding is a folder inside its label set. A set can hold several -- one per
instruction, or a subset of its ids -- each under its own name. The passages, their
`passages_sha` and the set's manifest are never touched.

```
data/dsld/labels/<set>/
├── manifest.json, passages.jsonl, annotations.jsonl      as today
└── embeddings/<name>/
    ├── manifest.json        written once, at creation; tracked in git
    ├── chunks/00000.npz     dsld_id int64 (n,), vector float32 (n, dim); written whole
    ├── chunks/00001.npz
    └── failed.jsonl         {"dsld_id": ..., "failed": "fits_context" | "error" | "empty"}
```

A chunk is written to a temporary name and renamed, so a crash never leaves a torn one.
`.gitignore` adds `data/dsld/labels/*/embeddings/*/chunks/` and
`data/dsld/labels/*/embeddings/*/*.jsonl`: vectors are rebuilt, not committed, as
passages are.

| Module | Function | Does |
|---|---|---|
| `paths.py` | `EMBEDDINGS_DIR`, `CHUNKS_DIR`, `FAILED_NAME`, `embedding_dir(label_set, name)` | locations; `MANIFEST_NAME` is reused |
| `store.py` | `create_embedding(label_set, name, manifest) -> Path` | makes the folder and writes its manifest; refuses an existing name, or a set that does not exist |
| | `append_chunk(label_set, name, ids, vectors)` | writes the next numbered chunk atomically |
| | `append_failed(label_set, name, rows)` | appends to `failed.jsonl` |
| `loader.py` | `read_embedding_manifest(label_set, name) -> dict` | the embedding's manifest |
| | `embedded_ids(label_set, name) -> set[int]` | ids in any chunk or in `failed.jsonl` |
| | `load_embedding(label_set, name) -> (ids, vectors)` | every chunk, concatenated, rows in the manifest's id order; refuses if the manifest's `passages_sha` is not the set's |

These are module-level functions, not `LabelsLoader` methods: an embedding grows while
a run writes it, so it is never cached.

**Joining a vector to its label** is one key and two reads, both from this loader:

```python
ids, vectors = loader.load_embedding("full214780-r4", name)    # (n,), (n, 1024)
_, labels = loader.load_flattened("full214780-r4")             # {dsld_id: Label}
labels[ids[0]].passage                                          # what vectors[0] was made from
```

A failed label has no row; its id is in `failed.jsonl`.

## The service (`services/label_embedding/service.py`)

`LabelEmbeddingService(controller=None)`, a context manager as purpose/risk. It loads
the `label_clustering` instruction and builds `EmbedController(instruction=...)`, which
builds the model; `controller` is the test seam.

`run(label_set, ids=None, name=None) -> Path`:

1. **`config_hash`** over what changes a vector: the instruction text, the model file
   and `max_attempts`. Not `n_servers`, `batch`, `ubatch` or `n_ctx`.
2. **The embedding.** `name` defaults to `label_clustering-<config_hash[:8]>`; `ids`
   (a subset of the label set, in the order given) requires an explicit `name`. A new
   name is created in the set with a manifest holding the set's `passages_sha`, the
   ids, the instruction and `config_hash`. An existing name is a resume: its ids come
   from the manifest, and a different `config_hash` or `passages_sha` is refused.
3. **The loop.** Labels whose id is not in `embedded_ids`, in manifest order, are streamed
   through the controller. An empty passage is recorded `empty` without being sent.
   Vectors are buffered and written as one chunk every 8,192; failures go to
   `failed.jsonl` as they settle. On `ModelUnavailable` the buffer is written and the
   error re-raised.
4. Returns the embedding's directory.

**Gold vectors live in `gold1000-r4`**, a label set written 2026-10-12: the
`gold1000-r1` ids in its order, with their `full214780-r4` passages (`passages_sha`
`0801f354cfbd8472`). It has no `annotations.jsonl`: renderer 4 lowercases and
normalises, so only 42 of the 4,281 gold evidence spans occur verbatim in its text.
The gold stays in `gold1000-r1`, joined by `dsld_id`. `run("gold1000-r4")` embeds it
(~20 s); instruction comparisons are runs on it under different instructions. The
full set's embedding is `run("full214780-r4")`.

Notebook 12's existing `data/dsld/embeddings/gold1000-r4-label_clustering-c30eabe4.npz`
predates this layout and is deleted once `run("gold1000-r4")` has replaced it.

## ADR 0002 v4

§8: "`embed_model.py` runs `n_servers` single-slot servers behind one interface,
set in `configs/embed.toml`"; its slots sentence goes. §5 notes a set's vectors live
with the label set, as its annotations do. "Structure today" adds
`services/label_embedding/`. No rule changes.

## Settings (`configs/embed.toml`)

`[model] n_servers = 3`, with the measured 1 / 2 / 3 / 4 server numbers in its
comment; `n_parallel` and its comment are removed, the slot-count numbers kept as the
reason. `_FALLBACKS` in the model loses `n_parallel` and gains `n_servers = 1`.
Nothing else changes.

## Acceptance

**Without a GPU (fakes):**

1. `EmbedModel(n_servers=3)` sends concurrent requests to the least-busy server;
   three requests in flight land on three different servers.
2. One server exiting mid-run raises `ModelUnavailable` on the next request, and
   `close()` stops all three.
3. `ready()` with less than `n_servers x min_vram_mib` free starts nothing.
4. The store refuses an existing name and a missing set; a chunk is never visible
   half-written; `embedded_ids` includes failed ids; `load_embedding` returns rows in
   manifest order, and the set's passages and manifest are unchanged after a run.
5. The service resumes: a second `run` on the same name embeds only the ids not done,
   and refuses a changed instruction.
6. `tests/test_layers.py` passes.

**On the GPU, run by the user:**

7. `ready()` with `n_servers = 3` passes; `nvidia-smi` shows three servers.
8. The same 20 texts through each server: cosine ≥ 0.9999 per text across servers.
9. 2,000 v4 passages through the service run at ≥ 1.6x the one-server rate
   (≥ ~17,000 tok/s).
