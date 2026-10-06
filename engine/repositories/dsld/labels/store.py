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
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

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
