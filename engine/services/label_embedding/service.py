"""Label embedding: a label set's passages -> controller -> vectors beside the set.

**The service only coordinates.** It loads the instruction, creates or checks
the embedding's manifest, and owns the one loop: each label's passage goes to
the controller as it pulls it, each vector into a buffer written as a chunk,
each failure to `failed.jsonl`. It does no batching, retrying or progress (the
controller's) and knows nothing of how the model is hosted, nor how many
servers it runs.

**It declares its own collaborators.** It is for the `label_clustering`
instruction, so it loads it and creates `EmbedController(instruction=...)`,
which creates the model. `controller` is the seam tests use; the instruction
hashed and recorded is then the controller's.

**An embedding is created whole, then resumed from its own files.** A new name
is created in the label set with a manifest holding the set's `passages_sha`,
the ids, the instruction and the `config_hash`. Running the same name again is
a resume: the ids come from the manifest, and a different `config_hash` or
`passages_sha` is refused -- the vectors would not be comparable. A label is
done when its id is in a chunk or in `failed.jsonl`.

`config_hash` covers what changes a vector: the instruction, the model file and
the attempt budget. Not `n_servers`, `batch`, `ubatch` or `n_ctx`: they change
speed, not vectors.

See `docs/specs/0008-label-embedding-service.md`.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine.controllers.embed_controller import EmbedController
from engine.repositories.dsld.labels import loader as label_sets
from engine.repositories.dsld.labels import paths as label_paths
from engine.repositories.dsld.labels import store as label_store
from engine.repositories.prompts import loader as prompts

log = logging.getLogger(__name__)

INSTRUCTION_ID = "label_clustering"

# Vectors buffered before they are written as one chunk: ~32 MiB at 1,024
# dimensions, and a few minutes of work lost at most to a crash.
_CHUNK_ROWS = 8192

# The failure name given to a label with an empty passage: nothing to embed.
EMPTY = "empty"


class LabelEmbeddingService:
    """Embeds DSLD label sets' passages under the `label_clustering` instruction."""

    def __init__(self, controller: EmbedController | None = None):
        """Creates the controller -- and through it the model -- unless given one."""
        if controller is None:
            controller = EmbedController(instruction=prompts.load_instruction(INSTRUCTION_ID))
        self.controller = controller

    def __enter__(self) -> "LabelEmbeddingService":
        return self

    def __exit__(self, *exc) -> None:
        self.controller.close()

    def run(self, label_set: str, ids: list[int] | None = None, name: str | None = None) -> Path:
        """Embed every label of the embedding that is not done yet; its directory.

        `ids` picks labels of `label_set`, in the order given, and then needs
        an explicit `name`; left out, the whole set is taken in its file order.
        `name` defaults to `label_clustering-<config_hash[:8]>`. A name that
        exists is resumed: its ids come from its manifest, and `ids`, if
        passed, must be the same.

        Only `ModelUnavailable` stops a run; whatever is buffered is written
        first, so a resume redoes nothing that finished.
        """
        # 1. What changes a vector, hashed. The model is reached through the
        # controller; a stand-in without a model file hashes by its name.
        instruction = self.controller.instruction
        model = self.controller.model
        vector_settings = {
            "instruction": instruction,
            "model": getattr(model, "model_path", type(model).__name__),
            "max_attempts": self.controller.max_attempts,
        }
        config_hash = hashlib.sha256(
            json.dumps(vector_settings, sort_keys=True).encode()
        ).hexdigest()[:16]

        # 2. The embedding: created with its ids, or reopened and checked.
        if ids is not None and name is None:
            raise ValueError("ids pick a subset of the set, so they need name=<embedding name>")
        name = name or f"{INSTRUCTION_ID}-{config_hash[:8]}"
        passages_sha, labels = label_sets.load_flattened(label_set)
        chosen = list(labels) if ids is None else [int(i) for i in ids]

        try:
            manifest = label_sets.read_embedding_manifest(label_set, name)
        except FileNotFoundError:
            manifest = None
        if manifest is not None:
            # A resume: the ids are the embedding's, and it is refused if it was
            # made another way or from other passages.
            if ids is not None and chosen != manifest["ids"]:
                raise ValueError(
                    f"embedding {name!r} of {label_set!r} exists with other ids; its ids come "
                    "from its manifest. Leave ids out to resume it, or pick a new name."
                )
            if manifest.get("config_hash") != config_hash:
                raise ValueError(
                    f"refusing to resume embedding {name!r} of {label_set!r}: its config_hash is "
                    f"{manifest.get('config_hash')!r} and this configuration's is {config_hash!r} "
                    "-- the instruction, model or attempt budget changed, so its vectors would "
                    "not be comparable. Pick a new name."
                )
            if manifest.get("passages_sha") != passages_sha:
                raise ValueError(
                    f"refusing to resume embedding {name!r} of {label_set!r}: it was made from "
                    f"passages_sha {manifest.get('passages_sha')}, and the set's is {passages_sha}."
                )
            chosen = manifest["ids"]
        else:
            # A new one: every id in the set, and named once.
            if len(set(chosen)) != len(chosen):
                raise ValueError("ids holds an id more than once")
            missing = [i for i in chosen if i not in labels]
            if missing:
                shown = ", ".join(str(i) for i in missing[:5])
                more = f" (+{len(missing) - 5})" if len(missing) > 5 else ""
                raise ValueError(
                    f"{len(missing)} of {len(chosen)} id(s) are not in label set "
                    f"{label_set!r}: {shown}{more}"
                )
            label_store.create_embedding(
                label_set,
                name,
                {
                    "passages_sha": passages_sha,
                    "instruction_id": INSTRUCTION_ID,
                    "instruction": instruction,
                    "model": vector_settings["model"],
                    "max_attempts": vector_settings["max_attempts"],
                    "config_hash": config_hash,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "ids": chosen,
                },
            )

        # 3. What is left, in manifest order. An empty passage is recorded
        # without being sent: there is nothing to embed.
        done = label_sets.embedded_ids(label_set, name)
        todo = [i for i in chosen if i not in done]
        empty = [i for i in todo if not labels[i].passage]
        label_store.append_failed(label_set, name, [{"dsld_id": i, "failed": EMPTY} for i in empty])
        work = [i for i in todo if labels[i].passage]
        n_already_done = len(chosen) - len(todo) + len(empty)

        # 4. The loop: outcome -> buffer or failed.jsonl. The buffer is written
        # every `_CHUNK_ROWS` vectors, at the end, and on the way out of any
        # error -- `ModelUnavailable` included, which is then re-raised.
        counts: collections.Counter[str] = collections.Counter({EMPTY: len(empty)})
        buffer_ids: list[int] = []
        buffer_vectors: list[np.ndarray] = []
        try:
            for dsld_id, outcome in self.controller.stream(
                ((i, labels[i].passage) for i in work),
                total=len(work),
                n_already_done=n_already_done,
            ):
                if outcome.ok:
                    counts["ok"] += 1
                    buffer_ids.append(dsld_id)
                    buffer_vectors.append(outcome.vector)
                    if len(buffer_ids) >= _CHUNK_ROWS:
                        label_store.append_chunk(label_set, name, buffer_ids, np.stack(buffer_vectors))
                        buffer_ids, buffer_vectors = [], []
                else:
                    counts[outcome.failed] += 1
                    label_store.append_failed(
                        label_set, name, [{"dsld_id": dsld_id, "failed": outcome.failed}]
                    )
        finally:
            if buffer_ids:
                label_store.append_chunk(label_set, name, buffer_ids, np.stack(buffer_vectors))

        d = label_paths.embedding_dir(label_set, name)
        log.info(
            "%s: %d labels this session (%s); %d done before",
            d, sum(counts.values()), dict(counts), n_already_done - len(empty),
        )
        return d
