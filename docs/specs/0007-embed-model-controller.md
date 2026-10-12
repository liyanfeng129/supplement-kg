# Embed model and controller

- **Status:** proposed
- **Date:** 2026-10-07 (revised 2026-10-12 after review)
- **Scope:** `engine/models/base_model.py` (new `Embedding`, `BaseEmbedModel`),
  new `engine/models/embed_model.py`, new `engine/controllers/embed_controller.py`,
  `engine/repositories/prompts/loader.py` (new `load_instruction`), new
  `engine/repositories/prompts/label_clustering/instruction.md`, new
  `configs/embed.toml`, `engine/config.py` (its path), new tests in
  `tests/test_embed_model.py`, `tests/test_embed_controller.py` and
  `tests/test_prompts.py` (or wherever the prompt loader is tested).

Governed by ADR 0002 (latest version). Modelled on the Qwen model and controller
(`docs/specs/0001-qwen-model-controller.md`); where this spec says "as Qwen", the
behaviour is copied from there and not restated.

Out of scope, for a follow-up spec: which text is embedded, choosing a subset from
the vectors (k-means), saving vectors through a repository, and how the cleaning
service drives the controller. Nothing in this spec knows what a label or a passage
is; the one instruction it ships is a starting wording the follow-up may revise.

## Why

A subset of the DSLD labels that covers as much of the domain as possible is chosen
by clustering labels on meaning. That needs a vector per flattened label: 214,780 v4
passages, about 53M tokens. This spec builds the model that turns text into vectors
and the controller that feeds it.

## Who owns what

| | Model (`EmbedModel`) | Controller (`EmbedController`) |
|---|---|---|
| **Owns** | Execution: the server, its slots; its own lifecycle and health; the instruction template | The model's existence; batching texts into requests, the in-flight budget, retry, progress |
| **Knows** | `n_parallel`, `n_ctx`, `ubatch`, the vector size, the end token | `batch`, `in_flight`, `max_attempts`, the instruction string |
| **Never knows** | What the text is or who is calling | The slot count; that a server exists; what a label is; where the instruction came from |

The controller imports `EmbedModel` concretely and builds it; `close()` closes it.
Passing a model in (`EmbedController(model)`) is the seam tests use. Nothing here
imports `services`; the controller does not import `repositories` either — the
service loads the instruction and passes the string.

## The shapes (`models/base_model.py`)

```python
@dataclass(frozen=True)
class Embedding:
    vectors: np.ndarray      # float32, shape (n, dim), each row L2-normalised
    latency_ms: int
    usage: Usage             # prompt_tokens = tokens embedded; completion_tokens = 0

class BaseEmbedModel(ABC):
    def ready(self) -> None: ...
    @abstractmethod
    def embed(self, texts: Sequence[str], instruction: str | None = None) -> Embedding: ...
    def fits(self, text: str, instruction: str | None = None) -> bool: ...   # default True
    def submit(self, texts: Sequence[str], instruction: str | None = None) -> Future[Embedding]: ...
        # default: embed, then resolve
    def close(self) -> None: ...
```

A separate class beside `BaseModel`, not a subclass: an embedding model has no
prompt and no generation. Rows are unit length, so cosine similarity is a dot
product. `Usage` is reused so token counts mean the same thing for both kinds of
model (ADR 0002 §4).

**The instruction is the caller's wording; the template is the model's.** How an
instruction is combined with a text is specific to each embedding model, as ChatML
is to Qwen, so the model applies it. `instruction=None` embeds the raw text.

## The model (`models/embed_model.py`)

**Weights:** `Qwen/Qwen3-Embedding-0.6B-GGUF`, **f16** (about 1.2 GB), 1024
dimensions, 32k context. f16 rather than Q8: quantising saves about 0.6 GB on a
24 GB card and costs quality; embedding is compute-bound and f16 runs on the tensor
cores. 4B was considered and rejected for now: about 8 GB, 2,560 dimensions and
several times slower, for a coarse coverage clustering. Switching is a change to
`embed.toml` and the dimension check only.

**Instruction template:** with an instruction, the text sent is
`Instruct: {instruction}\nQuery:{text}`, Qwen3-Embedding's documented format.

**Hosted on `llama-server`, privately, as Qwen:** one instance per process (`__new__`
returns the same object), a free local port, the server dies with its Python process,
no automatic restart, `ModelUnavailable` naming whatever holds the card. Started
with:

```
llama-server -m <model_path> --embedding --pooling last --embd-normalize 2 -ngl 999
             -np <n_parallel> -c <n_ctx> --kv-unified
             -ub <ubatch> -b <ubatch> --host 127.0.0.1 --port <port> --no-webui
```

- **`--pooling last`** is what Qwen3-Embedding is trained for. A wrong pooling mode
  gives well-formed vectors that mean nothing, with no error; `ready()` check 3
  catches it.
- **`--embd-normalize 2`** (L2) is the server's default, passed explicitly so the
  unit-length contract does not rest on a default. Python does not re-normalise;
  `ready()` check 2 confirms unit length.
- **Parallelism is slots, not instances.** One process loads the weights once; the
  server packs one sequence per slot into each forward pass, up to `ubatch` tokens.
  The average v4 passage is about 250 tokens (53M ÷ 214,780), so a pass is only full
  with many slots: 4 slots would carry about 1k tokens of an 8,192-token pass.
  Several processes on one GPU take turns on it rather than adding up. If a sweep
  ever shows two servers beating one, that change is inside this file only
  (ADR 0002 §8).
- **The KV cache is one shared pool (`--kv-unified`), not split per slot.** With a
  split cache every slot must reserve room for the longest passage: 32 × 8,192 ×
  112 KB is about 29 GB. With a unified pool of `n_ctx` tokens, most slots hold short
  passages and an occasional long one still fits.
- **A whole text must fit in one pass**: llama.cpp needs `ubatch` ≥ a text's tokens
  for embeddings (`input (…) is too large to process`). That is the one per-text
  limit. The longest v4 passage is about 4.5k tokens; 8,192 covers it with the
  instruction.

**End token.** Qwen3-Embedding's last-token pooling expects `<|endoftext|>` at the
end of every text. The server appends an end token only when the GGUF sets
`add_eos_token`. `ready()` tokenises a probe through `/tokenize`; if the last token
is not `<|endoftext|>`, `embed` appends `<|endoftext|>` to every (templated) text
from then on. `fits` counts that token.

**Verbs:**

| verb | what it does |
|---|---|
| `ready()` | Idempotent. Refuses unless the card has `min_vram_mib` free, then starts the server, settles the end token, and runs the three checks below. Refuses — does not restart — a server it started that has since exited. |
| `embed(texts, instruction)` | Templates each text, appends the end token if needed, makes one blocking `POST /v1/embeddings` with the list; returns an `Embedding` with rows in input order and the server's token count. |
| `fits(text, instruction)` | On the templated text: `len(templated) + 1 ≤ ubatch` answers `True` with no call (a token covers at least one character; +1 for the end token). Otherwise one `/tokenize` call compares the count against `ubatch`. |
| `submit(texts, instruction)` | Queues one `embed` call on a pool of `n_parallel` workers; returns a `Future[Embedding]` at once. Never blocks, never refuses. |
| `close()` | Cancels queued calls, stops the server, frees the card. Idempotent. |

**`ready()` checks**, in order; any failure stops the server and raises
`ModelUnavailable`:

1. **The weights reached the GPU:** the VRAM the load took is at least 90% of the
   weights file's size, as Qwen. Catches the silent CPU fallback.
2. **Vectors are well-formed:** embedding a short text gives one row of 1024
   dimensions, unit length, no NaN.
3. **Similarity is ordered sensibly:** two near-identical supplement passages
   (fixed in the module) are more similar to each other than either is to an
   unrelated text — once without an instruction and once with a test instruction
   fixed in the module. Weak on purpose: it catches broken pooling, a missing end
   token or a broken template, not a subtly worse model.

**Failed requests** are handled as Qwen: on a failed `/v1/embeddings` call the model
checks its own health. Process exited, a CUDA or out-of-memory failure, or `/health`
not answering ok within 5 s raises `ModelUnavailable`; a healthy server raises the
request's own error, which the controller retries.

## The controller (`controllers/embed_controller.py`)

**Construction:** `EmbedController(model=None, instruction=None, batch=None,
in_flight=None, max_attempts=None)`; builds `EmbedModel()` when no model is given;
`batch`, `in_flight` and `max_attempts` come from `[controller]` in `embed.toml`
unless passed. `instruction` is fixed for the controller's life and passed on every
`fits` and `submit`: vectors made under different instructions are not comparable,
so one run uses one. `close()` closes the model.

**Two entry points:**

- `stream(work, total=None, n_already_done=0, on_progress=None)` takes a lazy
  iterable of `(key, text)` and yields `(key, Outcome)` as each settles, in
  completion order. The key is opaque.
- `handle(texts, on_progress=None) -> np.ndarray` is `stream` over an in-memory
  list, returned as one `(n, dim)` array in input order. If any text is not `ok`,
  it raises `ValueError` naming the failed indices and why: an array has no place
  for a missing row. For tests, notebooks and small sets.

```python
TOO_LONG = "fits_context"   # as Qwen
ERROR = "error"             # as Qwen

@dataclass(frozen=True)
class Outcome:
    vector: np.ndarray | None   # float32, (dim,); None unless ok
    ok: bool
    attempts: int
    failed: str | None          # TOO_LONG or ERROR; None when ok
```

`failed` is one name, not Qwen's list: an embedding fails at most one way.

**The loop:**

- **Empty text is a caller bug.** A text that is empty raises `ValueError` at once.
  A flattened v4 label always keeps its `fullName`, so none should be empty; one
  that reached the server would fail its whole batch with a misleading reason.
- **Fit first.** Each text pulled from `work` is checked with `model.fits`; one that
  does not fit is handed back at once as `TOO_LONG` and never reaches the model.
  Never truncated.
- **Batch.** Fitting texts are grouped, in pull order, into requests of up to
  `batch` texts. A request is submitted when it is full or `work` is exhausted.
- **In flight.** At most `in_flight` requests are submitted but not yet settled. Each
  settled request makes room to pull and submit one more. The server splits a request
  into one task per text across all slots, so what must exceed the slot count is the
  queued *texts*: `in_flight × batch > n_parallel`, so a freed slot always finds a
  text waiting. The controller never reads `n_parallel`; the ratio is kept by hand
  through the comment in `embed.toml`.
- **Retry.** A request that raises anything but `ModelUnavailable` is re-sent as it
  was, as a whole batch, until `max_attempts`; then each of its texts settles as
  `ERROR` and the stream goes on. Every text in a batch carries that batch's
  `attempts`.
- **Stop.** `ModelUnavailable` ends the stream: outstanding requests are cancelled
  and it propagates.
- **Progress.** Counted per text: done of total, tokens per second over the wall
  clock, ETA over a recent window, as Qwen's meter. A resumed run passes
  `n_already_done`. Relayed to `on_progress`; nothing is printed.

There are no checks and no parser: an embedding has no text to validate. Shape and
sanity are the model's to guarantee, in `ready()`.

## The instruction (`repositories/prompts/`)

The wording is workflow data, so it lives in the prompt repository beside the Qwen
prompts. The service loads it and hands the string to the controller; neither the
controller nor the model knows it came from a file.

- **`label_clustering/instruction.md`** holds one sentence, the starting wording:

  ```
  Identify the type of dietary supplement product described by this label, its main ingredients and its intended use
  ```

  The follow-up spec may revise it.
- **`load_instruction(prompt_id) -> str`** in `loader.py` reads
  `<id>/instruction.md`, stripped of its trailing newline. An instruction folder has
  no Python files. An unknown id raises `KeyError` listing the known instruction ids,
  as `load` does. `load` itself is unchanged: it still requires `prompt.md`,
  `checks.py`, `parse.py`, `STOP` and `MAX_TOKENS`, none of which mean anything for
  an embedding.

## Where each piece lives

Only the functions below are written. The server start, health and stop logic is
copied from `qwen_model.py`, not shared: a common helper is a later refactor, once
two working copies show what is really common. Likewise the progress meter is
copied from `qwen_controller.py`.

| File | Function | Does |
|---|---|---|
| `models/base_model.py` | `Embedding`, `BaseEmbedModel` | the shapes above |
| `models/embed_model.py` | `EmbedModel.__new__` | one instance per process, settings as Qwen (a changed setting restarts an idle server, is refused mid-run) |
| | `ready`, `embed`, `fits`, `submit`, `close` | the verbs above; the end-token probe, the template and the three checks are inline |
| | `_request(path, body, timeout)` | one HTTP call to the server; the health check on failure, as Qwen's |
| `controllers/embed_controller.py` | `Outcome`, `TOO_LONG`, `ERROR` | the shape above |
| | `EmbedController.__init__`, `close` | construction |
| | `stream` | the loop above, inline |
| | `handle` | `stream` collected into an array |
| | `_Meter` | progress, copied from Qwen's |
| `repositories/prompts/loader.py` | `load_instruction` | reads `<id>/instruction.md` |
| `repositories/prompts/label_clustering/` | `instruction.md` | the starting wording |

## Settings (`configs/embed.toml`)

Committed and diffable, as `qwen.toml`. An explicit constructor argument beats the
file; a missing or malformed file falls back to the defaults in the code.

| Table | Key | Starting value | Note |
|---|---|---|---|
| `[model]` | `model_path` | the f16 GGUF in the HF cache | |
| | `server_path` | the same `llama-server` as Qwen | |
| | `n_parallel` | 32 | provisional: acceptance 13 |
| | `n_ctx` | 32768 | the shared KV pool; provisional: acceptance 13 |
| | `ubatch` | 8192 | the per-text limit; ≥ the longest templated passage |
| | `min_vram_mib` | 8000 | see below |
| `[controller]` | `batch` | 32 | provisional: acceptance 13 |
| | `in_flight` | 4 | `in_flight × batch` must exceed `n_parallel` |
| | `max_attempts` | 3 | |

**`n_ctx` starts at 4 × `ubatch`.** A text cannot be split across passes, so one pass
holds at most `ubatch` tokens and a pool of `ubatch` might suffice. Whether the
server frees a slot's KV as soon as its embedding returns, or only when the slot is
reused, was not confirmed; if the latter, a small pool fills and requests fail with
no KV slot. Acceptance 11 decides.

**`min_vram_mib` is 8000.** Qwen3-Embedding-0.6B has 28 layers and 8 KV heads of 128
dimensions, about 112 KB per token at f16, so a 32,768-token pool is about 3.7 GB,
plus 1.2 GB of weights and the compute buffer for an 8,192-token pass. Measure the
real peak in acceptance 13 and set this from it. Qwen (at least 21.5 GB) and this
model do not fit on the card together; embedding and extraction run as separate
phases.

## Acceptance

**With a fake model, no GPU:**

1. `handle` returns vectors in input order although batches settle out of order.
2. `stream` never has more than `in_flight` requests outstanding, and pulls `work`
   lazily.
3. A failed request is retried as a batch up to `max_attempts`, then its texts
   settle as `ERROR`; the stream goes on.
4. `ModelUnavailable` ends the stream.
5. A text that does not fit settles as `TOO_LONG` with `vector=None` and never
   reaches the model.
6. An empty text raises `ValueError`.
7. The controller's `instruction` reaches every `fits` and `submit` call.
8. `on_progress` reports done of total, tokens per second and an ETA.
9. `fits` makes no `/tokenize` call for a templated text with
   `len(templated) + 1 ≤ ubatch`.
10. `load_instruction("label_clustering")` returns the sentence; an unknown id raises
    `KeyError` naming the known ids. `tests/test_layers.py` passes.

**On the GPU, run by the user:**

11. `ready()` settles the end token and passes all three checks.
12. The same text gives the same vector whatever batch it is in, within cosine
    ≥ 0.9999 (llama.cpp floating point can move slightly with batch shape).
13. **Throughput sweep** on 2,000 real v4 passages with the `label_clustering`
    instruction: `n_parallel` ∈ {8, 16, 32, 64} × `batch` ∈ {16, 32, 64}, and
    `n_ctx` ∈ {8192, 32768} at the winner, load excluded, recording tokens/s, peak
    VRAM and any KV-slot errors in the server log. Keep `in_flight × batch >
    n_parallel` throughout. The winner and the measured peak go into `embed.toml`
    with the numbers in its comments; the smaller `n_ctx` only if it ran clean. At
    53M tokens, 20k tokens/s is about 44 minutes for the full set.
