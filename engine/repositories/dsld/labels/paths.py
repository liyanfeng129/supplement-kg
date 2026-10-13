"""Where label sets live. Locations only: nothing here reads or writes.

A label set is a directory under `LABELS_ROOT`, named for good when it is
written: `labels/<name>/passages.jsonl` and its `manifest.json`, plus
`annotations.jsonl` when the set is gold or silver, and any embeddings of it
under `embeddings/<name>/`. A test redirects label
sets by patching `LABELS_ROOT` here, and nowhere else: it is read at call
time, never bound as a default argument.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

LABELS_ROOT = REPO_ROOT / "data" / "dsld" / "labels"

PASSAGES_NAME = "passages.jsonl"
MANIFEST_NAME = "manifest.json"
ANNOTATIONS_NAME = "annotations.jsonl"

# An embedding of a set: `<set>/embeddings/<name>/`, holding its own
# `manifest.json` (MANIFEST_NAME), `chunks/00000.npz`, ... and `failed.jsonl`.
EMBEDDINGS_DIR = "embeddings"
CHUNKS_DIR = "chunks"
FAILED_NAME = "failed.jsonl"


def set_dir(name: str, root: str | Path | None = None) -> Path:
    """The directory of label set `name`. No existence check."""
    return Path(root or LABELS_ROOT) / name


def embedding_dir(label_set: str, name: str) -> Path:
    """The directory of embedding `name` of label set `label_set`. No existence check."""
    return set_dir(label_set) / EMBEDDINGS_DIR / name
