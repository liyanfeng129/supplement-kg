"""The entry to save label sets: `write_label_set`, as JSONL under `paths.LABELS_ROOT`.

**A label set is written once under a name that says what it is.** The
cleaning service saves its flattened labels here: a passages file plus a
manifest that records, among what the set is, `passages_sha` -- a hash of the
passages. A name means one set of passages for good: saving the same passages
again writes nothing, and saving different ones under a used name is refused.
A set may also hold the annotations made on it, which is what makes it a gold
or silver set. A run refers to its labels by name and `passages_sha` and never
copies them, so a result can always be joined back to the passage it was read
from.

**An embedding of a set lives inside it**, beside the passages it was made
from, under `embeddings/<name>/`: `create_embedding` writes its manifest once,
`append_chunk` adds vectors, `append_failed` adds the ids that have none. The
set's own files are never touched. A chunk is written whole or not at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

from engine.repositories.dsld.labels import paths
from engine.repositories.dsld.labels.schema import Label


def write_label_set(
    name: str, rows: list[dict], manifest: dict, root: str | Path | None = None
) -> str:
    """Save flattened labels as label set `name`; return its `passages_sha`.

    `passages_sha` is the first 16 hex digits of the sha256 of
    the passages file as written. It goes into the manifest beside `name`, so
    the manifest passed in describes the set and need not carry either.

    A name is one set of passages for good. If `name` exists with the same
    `passages_sha`, nothing is written and its manifest is kept; if it exists
    with a different one, this refuses -- changed passages need a new name, or
    every run that read the old ones would silently point at the new.

    Every row must read as a `Label` and ids must be unique: a set is looked up
    by id, and a duplicate would silently shadow the other row. A new set is
    written into a hidden temporary directory and renamed into place, so a
    half-written set never exists under a name.
    """
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"{name!r} cannot name a label set directory")
    seen: set[int] = set()
    for n, r in enumerate(rows, 1):
        i = Label.from_dict(r).dsld_id
        if i in seen:
            raise ValueError(f"row {n}: dsld_id {i} appears twice; a label set holds each id once")
        seen.add(i)

    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    passages_sha = hashlib.sha256(data).hexdigest()[:16]
    d = paths.set_dir(name, root)
    if d.exists():
        existing = json.loads((d / paths.MANIFEST_NAME).read_text()).get("passages_sha")
        if existing != passages_sha:
            raise ValueError(
                f"label set {name!r} already exists with passages_sha {existing}, and these "
                f"passages hash to {passages_sha}. Changed passages need a new name."
            )
        return passages_sha

    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.parent / f".{name}-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        (tmp / paths.PASSAGES_NAME).write_bytes(data)
        (tmp / paths.MANIFEST_NAME).write_text(
            json.dumps(
                {"name": name, **manifest, "name": name, "passages_sha": passages_sha},
                indent=2,
                ensure_ascii=False,
            )
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        os.rename(tmp, d)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not d.exists():
            raise
        # A concurrent write won the race; the check above decides again.
        return write_label_set(name, rows, manifest, root)
    return passages_sha


def create_embedding(label_set: str, name: str, manifest: dict) -> Path:
    """Make embedding `name` in label set `label_set` and write its manifest; its directory.

    The manifest is written once, here, and never changed: a resumed run reads
    it back. A used name is refused -- a different embedding would land among
    the old one's vectors -- and so is a set that does not exist. The folder is
    written under a hidden temporary name and renamed into place, so a
    manifest-less embedding never exists under a name.
    """
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"{name!r} cannot name an embedding directory")
    set_d = paths.set_dir(label_set)
    if not (set_d / paths.PASSAGES_NAME).exists():
        raise FileNotFoundError(
            f"no label set {label_set!r} at {set_d}: an embedding is made of a saved set."
        )
    d = paths.embedding_dir(label_set, name)
    if d.exists():
        raise ValueError(
            f"embedding {name!r} of label set {label_set!r} already exists at {d}. "
            "Resume it, or pick a new name."
        )

    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.parent / f".{name}-{uuid.uuid4().hex}"
    (tmp / paths.CHUNKS_DIR).mkdir(parents=True)
    try:
        (tmp / paths.MANIFEST_NAME).write_text(
            json.dumps({**manifest, "name": name, "label_set": label_set}, indent=2, ensure_ascii=False)
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        os.rename(tmp, d)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if d.exists():
            # A concurrent create won the race.
            raise ValueError(
                f"embedding {name!r} of label set {label_set!r} already exists at {d}."
            ) from None
        raise
    return d


def append_chunk(label_set: str, name: str, ids: Sequence[int] | np.ndarray, vectors: np.ndarray) -> None:
    """Write `ids` (n,) and their `vectors` (n, dim) as the embedding's next chunk.

    Chunks are numbered from `00000.npz`. One is written to a hidden temporary
    file and renamed, so a crash never leaves a torn chunk for a reader to
    find. Writing no rows writes nothing.
    """
    ids = np.asarray(ids, dtype=np.int64)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(ids) != len(vectors):
        raise ValueError(f"{len(ids)} ids for vectors of shape {vectors.shape}")
    if not len(ids):
        return

    chunks = paths.embedding_dir(label_set, name) / paths.CHUNKS_DIR
    if not chunks.is_dir():
        raise FileNotFoundError(f"no embedding {name!r} of label set {label_set!r} at {chunks.parent}")
    numbers = [int(p.stem) for p in chunks.glob("*.npz") if p.stem.isdigit()]
    target = chunks / f"{max(numbers, default=-1) + 1:05d}.npz"
    tmp = chunks / f".{target.stem}-{uuid.uuid4().hex}"
    try:
        with tmp.open("wb") as fh:
            np.savez(fh, dsld_id=ids, vector=vectors)
        os.rename(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def append_failed(label_set: str, name: str, rows: list[dict]) -> None:
    """Append `{"dsld_id": ..., "failed": ...}` rows to the embedding's `failed.jsonl`.

    A line torn by an earlier crash is cut first, so it never ends up between
    two whole ones: it costs that one id, which the next run embeds again.
    """
    if not rows:
        return
    d = paths.embedding_dir(label_set, name)
    if not d.is_dir():
        raise FileNotFoundError(f"no embedding {name!r} of label set {label_set!r} at {d}")
    target = d / paths.FAILED_NAME
    if target.exists():
        data = target.read_bytes()
        if data and not data.endswith(b"\n"):
            with target.open("r+b") as fh:
                fh.truncate(data.rfind(b"\n") + 1)
    with target.open("a") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
