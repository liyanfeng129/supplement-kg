"""Profile a finished purpose/risk run: each claim's type, each label's group, the counts.

Deterministic, over every claim, no model and no judge (spec 0011.01). This
module is the one definition of the four faithfulness types. Against the
passage the model read, where a segment is one line and words are
`str.split()`, compared as sets, the first match wins:

- `verbatim`: the claim is a substring of the passage;
- `stitched`: one segment holds every word of the claim;
- `spread`: the passage holds every word, but no single segment does;
- `invented`: some word of the claim is not in the passage.

A type says where a claim's words come from, not whether its meaning
survived. A label's group comes from its row alone: `both`, `purpose_only`,
`risk_only` or `neither`.

`profile()` is the pure core; `build_profile()` reads the run and its label
set, checks they belong together, computes, and saves the profile beside the
run. `VERSION` is bumped by hand when a type or group rule changes; a stored
profile of another version is the reader's to refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

from engine.repositories.dsld.labels import loader as labels_loader
from engine.repositories.dsld.labels.schema import Label
from engine.repositories.dsld.runs import loader as runs_loader
from engine.repositories.dsld.runs import paths as runs_paths
from engine.repositories.dsld.runs import store as runs_store
from engine.repositories.dsld.runs.schema import Group, Kind, ProfiledLabel, Type
from engine.services.extraction_eval.schema import Profile

VERSION = 1


def profile(rows: list[dict], labels: dict[int, Label]) -> Profile:
    """Type every claim and group every label of a run's rows. Pure, no I/O.

    `labels` is the label set the run read, by id. A row whose `status` is not
    `ok` gets no group and is only listed in `failed_ids`. Raises on a row
    whose id is not in `labels` (the wrong set), an id with two rows, and a
    claim whose text is empty after stripping, which would otherwise count as
    verbatim.
    """
    claim_counts = {
        (g, k, t): 0 for g in get_args(Group) for k in get_args(Kind) for t in get_args(Type)
    }
    label_counts = {g: 0 for g in get_args(Group)}
    profiled: list[ProfiledLabel] = []
    failed_ids: list[int] = []
    seen: set[int] = set()

    for row in rows:
        # One row per label, from the label set the run read.
        dsld_id = int(row["dsld_id"])
        if dsld_id in seen:
            raise ValueError(f"dsld_id {dsld_id} has two rows; a run holds one row per label")
        seen.add(dsld_id)
        if dsld_id not in labels:
            raise ValueError(
                f"dsld_id {dsld_id} is not in the label set: the wrong label set was passed"
            )
        if row.get("status") != "ok":
            failed_ids.append(dsld_id)
            continue

        # Type each claim against the passage, first match wins.
        passage = labels[dsld_id].passage
        segments = [set(s.split()) for s in passage.split("\n")]
        words = set(passage.split())
        types: dict[Kind, list[Type]] = {}
        for kind in get_args(Kind):
            types[kind] = []
            for i, item in enumerate(row[kind]):
                text = item["text"]
                if not text.strip():
                    raise ValueError(f"claim:{dsld_id}:{kind}:{i} has an empty text")
                claim_words = text.split()
                if text in passage:
                    types[kind].append("verbatim")
                elif any(all(w in s for w in claim_words) for s in segments):
                    types[kind].append("stitched")
                elif all(w in words for w in claim_words):
                    types[kind].append("spread")
                else:
                    types[kind].append("invented")

        # Group from the row alone, whatever its claims' types.
        if types["purpose"] and types["risk"]:
            group: Group = "both"
        elif types["purpose"]:
            group = "purpose_only"
        elif types["risk"]:
            group = "risk_only"
        else:
            group = "neither"

        label_counts[group] += 1
        for kind, kind_types in types.items():
            for t in kind_types:
                claim_counts[(group, kind, t)] += 1
        profiled.append(ProfiledLabel(dsld_id, group, types["purpose"], types["risk"]))

    return Profile(profiled, claim_counts, label_counts, failed_ids)


def build_profile(run_dir: str | Path) -> Profile:
    """Profile a finished run and save it in the run's folder; return the profile.

    Refuses before scanning if the run already has a profile file, and if the
    label set named in the run's manifest is not the one the run read (its
    `passages_sha` differs). The saved manifest pins the run's `config_hash`,
    `passages_sha` and, through the store, its rows, plus `VERSION`.
    """
    # Refuse up front: the store would refuse too, but only after the scan.
    for p in (
        runs_paths.rows_path(run_dir, runs_paths.PROFILE_NAME),
        runs_paths.manifest_path(run_dir, runs_paths.PROFILE_MANIFEST_NAME),
    ):
        if p.exists():
            raise FileExistsError(
                f"{p} exists: run {run_dir} is already profiled. Read it with "
                f"runs.loader.read_profile(), or delete both profile files to re-profile."
            )

    # Read the run and the label set it ran on, and check they belong together.
    rows = runs_loader.read_rows(run_dir)
    manifest = runs_loader.read_manifest(run_dir)
    sha, labels = labels_loader.load_flattened(manifest["label_set"])
    if sha != manifest["passages_sha"]:
        raise ValueError(
            f"label set {manifest['label_set']!r} has passages_sha {sha}, but run {run_dir} "
            f"read {manifest['passages_sha']}: a profile of the wrong label set"
        )

    prof = profile(rows, labels)

    # Save, with the cube nested for JSON, which has no tuple keys.
    claim_counts: dict = {}
    for (g, k, t), n in prof.claim_counts.items():
        claim_counts.setdefault(g, {}).setdefault(k, {})[t] = n
    runs_store.write_profile(
        run_dir,
        prof.labels,
        {
            "run": Path(run_dir).name,
            "config_hash": manifest["config_hash"],
            "passages_sha": manifest["passages_sha"],
            "classifier_version": VERSION,
            "failed_ids": prof.failed_ids,
            "claim_counts": claim_counts,
            "label_counts": prof.label_counts,
        },
    )
    return prof
