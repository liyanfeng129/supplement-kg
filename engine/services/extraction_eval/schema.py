"""What the extraction evaluation hands back.

`ProfiledLabel` and the `Kind` / `Type` / `Group` names live with the runs
repository, which stores and returns them; the service only adds the shapes
it alone produces.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.repositories.dsld.runs.schema import Group, Kind, ProfiledLabel, Type


@dataclass(frozen=True)
class Profile:
    """A run's profile: every ok label with its group and claim types, and the counts.

    `claim_counts` is the full group × kind × type cube, every cell present,
    zeros included; `label_counts` has every group. Any marginal is a sum over
    them, left to the frontend.
    """

    labels: list[ProfiledLabel]  # run order, ok rows only
    claim_counts: dict[tuple[Group, Kind, Type], int]
    label_counts: dict[Group, int]
    failed_ids: list[int]
