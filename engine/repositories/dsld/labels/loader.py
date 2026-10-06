"""The entry to read label sets: `load_flattened(name) -> (passages_sha, {dsld_id: Label})`.

A label set is what the cleaning service saved: id, name and one passage each.
What an LLM is run *over*. Callers ask this module for it and never open the
files themselves; where a set lives is `paths.py`'s business.

**A label set is read whole, by name.** A set was chosen and written by the
cleaning service, so there is nothing to select here: which labels a run
does, and in what order, is the run's business. The set's `passages_sha`
comes back with the labels -- checked against the file, so a set edited in
place is refused -- and a run records it, so a resume can tell whether the
passages it started on are still the ones under that name.

**One loader instance.** A label set does not change under its name, so it is
parsed once per process and shared. `reset()` exists for tests.

A set's `annotations.jsonl` is not read here yet; a loader for it comes when
scoring is built.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from engine.repositories.dsld.labels import paths
from engine.repositories.dsld.labels.schema import Label


class LabelsLoader:
    """The single entry to read label sets.

    One instance per process. It holds the parsed label sets, keyed by name:
    a set does not change under its name, so a cached one is not stale.
    """

    _instance: "LabelsLoader | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "LabelsLoader":
        # Double-checked under a lock: the controllers layer may ask for the
        # loader from more than one worker thread, and two instances would
        # mean two caches and two answers to the same question.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._flattened: dict[str, tuple[str, dict[int, Label]]] = {}

    @classmethod
    def reset(cls) -> None:
        """Drop the instance and everything it has cached.

        For tests.
        """
        with cls._lock:
            cls._instance = None

    def load_flattened(self, name: str) -> tuple[str, dict[int, Label]]:
        """A whole label set, and its `passages_sha`, as the cleaning service saved it.

        The labels come back keyed by id, in the set's file order. The dict is
        a copy, so a caller that edits it damages nothing; `Label` is frozen.

        The passages are hashed on read and must match the manifest's
        `passages_sha`: a set edited in place would otherwise be read as the
        set a run started on.
        """
        cached = self._flattened.get(name)
        if cached is None:
            d = paths.set_dir(name)
            rows_p = d / paths.PASSAGES_NAME
            if not rows_p.exists():
                raise FileNotFoundError(
                    f"no label set {name!r} at {d}. Label sets are saved by the "
                    f"cleaning service's save()."
                )
            manifest = json.loads((d / paths.MANIFEST_NAME).read_text())
            passages_sha = hashlib.sha256(rows_p.read_bytes()).hexdigest()[:16]
            if passages_sha != manifest.get("passages_sha"):
                raise ValueError(
                    f"label set {name!r}: {paths.PASSAGES_NAME} hashes to {passages_sha} but its "
                    f"manifest says {manifest.get('passages_sha')} -- it was edited in "
                    f"place. Save changed passages as a new set."
                )
            labels = {}
            for record in _read_jsonl(rows_p):
                label = Label.from_dict(record)
                labels[label.dsld_id] = label
            cached = (passages_sha, labels)
            self._flattened[name] = cached
        return cached[0], dict(cached[1])


def _read_jsonl(src: Path) -> list[dict]:
    """Parse a JSONL file, naming the line number of a bad record.

    A corpus file is edited by hand often enough that "line 412" is the
    difference between a one-minute fix and a bisect.
    """
    out = []
    with src.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{src}:{n}: {exc}") from exc
    return out


def loader() -> LabelsLoader:
    """The one loader. Cheap to call; every call returns the same instance."""
    return LabelsLoader()


# Module-level entries. The class is the thing, but a caller reading labels
# should not have to know it exists -- `loader.load_flattened()` is the entry,
# and that it is backed by a shared instance is this file's business.


def load_flattened(name: str) -> tuple[str, dict[int, Label]]:
    """A whole label set and its `passages_sha`. See `LabelsLoader.load_flattened`."""
    return loader().load_flattened(name)


def reset() -> None:
    """Drop the loader and its caches. See `LabelsLoader.reset`."""
    LabelsLoader.reset()
