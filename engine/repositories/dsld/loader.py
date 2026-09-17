"""The entry to read DSLD labels.

Being *the entry* is the whole job: callers ask this module for labels and
never open the file themselves. Where the labels physically live -- a frozen
pilot JSONL today, a database or the DSLD API later -- is an implementation
detail behind `load()`, and moving them changes this file and nothing above it.

**Resume lives here**, because resume is a question about what is left to read:
the store knows which ids are done, so the loader consults it and hands back
only the rest. A caller that wants everything simply does not pass a run.

**A selection is returned, not a bare list.** The old loader returned records
and printed the counts it had dropped, which meant a notebook calling it either
scraped stdout or guessed. `Selection` carries the numbers that were previously
only printed -- how many matched, how many were already done, which requested
ids do not exist -- so the caller can report them however it likes. That is the
"say what is in and out" rule from ADR 0002 §4 applied to the one layer that
knows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from engine.repositories.dsld import store
from engine.repositories.dsld.schema import Label

# The frozen pilot passages. The gold set was labelled against these exact
# strings, so a run that re-renders instead of reading them is not comparable
# to the gold set -- see `schema.Label`.
DEFAULT_RECORDS = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "dsld"
    / "runs"
    / "20260908T125504Z"
    / "raw"
    / "pilot-1000.jsonl"
)


@dataclass(frozen=True)
class Selection:
    """Labels to process, and an account of everything left out.

    The account is the point. `labels` alone cannot distinguish "you asked for
    500 and the file holds 200" from "you asked for 500 and 300 were already
    done", and those call for opposite responses from the caller.
    """

    labels: list[Label]
    n_available: int = 0
    n_already_done: int = 0
    missing_ids: list[int] = field(default_factory=list)
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    @property
    def ids(self) -> list[int]:
        return [lab.dsld_id for lab in self.labels]

    def summary(self) -> str:
        """One line a frontend can print as-is, or ignore in favour of the fields."""
        parts = [f"{len(self.labels)} to do of {self.n_available} available"]
        if self.n_already_done:
            parts.append(f"{self.n_already_done} already done")
        if self.missing_ids:
            shown = ", ".join(str(i) for i in self.missing_ids[:5])
            more = f" (+{len(self.missing_ids) - 5})" if len(self.missing_ids) > 5 else ""
            parts.append(f"{len(self.missing_ids)} requested id(s) not in source: {shown}{more}")
        return "; ".join(parts)


def read_all(path: str | Path | None = None) -> list[Label]:
    """Every label in the source, in file order.

    Separate from `load()` so that counting or inspecting the corpus does not
    require constructing a selection, and so `load()` has one obvious place to
    get its universe from.
    """
    src = Path(path or DEFAULT_RECORDS)
    if not src.exists():
        raise FileNotFoundError(
            f"no label records at {src}. The pilot passages are gitignored "
            f"(raw/ is not committed); re-run the pilot or point --records elsewhere."
        )
    labels = []
    with src.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                labels.append(Label.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{src}:{n}: {exc}") from exc
    return labels


def load(
    path: str | Path | None = None,
    ids: list[int] | None = None,
    limit: int | None = None,
    resume_from: str | Path | None = None,
) -> Selection:
    """Labels to process, with an account of what was left out and why.

    `ids` selects specific labels; ids absent from the source are reported in
    `Selection.missing_ids` rather than silently yielding a shorter list, since
    a typo'd id and a genuinely absent one look identical in a bare result.

    `limit` applies after id selection and after the resume filter, so
    `limit=10` means ten labels that will actually be processed -- not ten
    candidates of which some are already done. The old loader applied it before
    the resume filter, where `--limit 10` on a resumed run could hand back
    nothing at all and look like completion.

    `resume_from` is a run directory. Labels already present in its rows are
    dropped and counted. It is explicit and never inferred: appending to an
    existing run is a deliberate act.
    """
    src = Path(path or DEFAULT_RECORDS)
    labels = read_all(src)
    n_available = len(labels)

    missing: list[int] = []
    if ids is not None:
        wanted = [int(i) for i in ids]
        by_id = {lab.dsld_id: lab for lab in labels}
        missing = [i for i in wanted if i not in by_id]
        # Caller order, not file order: an explicit id list is usually a
        # priority order, and silently re-sorting it hides that it was ignored.
        labels = [by_id[i] for i in wanted if i in by_id]

    n_done = 0
    if resume_from is not None:
        done = store.done_ids(resume_from)
        if done:
            before = len(labels)
            labels = [lab for lab in labels if lab.dsld_id not in done]
            n_done = before - len(labels)

    if limit is not None:
        labels = labels[:limit]

    return Selection(
        labels=labels,
        n_available=n_available,
        n_already_done=n_done,
        missing_ids=missing,
        source=src,
    )
