"""What the extraction evaluation hands back.

`ProfiledLabel` and the `Kind` / `Type` / `Group` names live with the runs
repository, which stores and returns them; the service only adds the shapes
it alone produces.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.repositories.dsld.runs.schema import (
    Group,
    Kind,
    ProfiledLabel,
    SampledLabel,
    StratumSize,
    Type,
)


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


@dataclass(frozen=True)
class Quota:
    """How many labels of one stratum each set takes, before "all" is applied."""

    pilot: int
    pilot_human: int  # ≤ pilot
    large: int
    human: int  # ≤ large


@dataclass(frozen=True)
class Sample:
    """The pilot, the large set (human set flagged) and every stratum's sizes."""

    seed: int
    pilot: list[SampledLabel]
    large: list[SampledLabel]  # stratum order, then rank; human set = in_human
    sizes: list[StratumSize]  # all 10 strata


@dataclass(frozen=True)
class SheetClaim:
    """One claim as the human sheet shows it: its id, kind and text, nothing about how it was made."""

    claim_id: str  # claim:<dsld_id>:<kind>:<i>
    kind: Kind
    text: str


@dataclass(frozen=True)
class SheetCase:
    """One label as the human sheet shows it: the passage and its claims, blind (spec 0011.04).

    No stratum, group, claim type or rank: any of them would tell the human
    where the sample looks harder.
    """

    case_id: str  # label:<dsld_id>
    dsld_id: int
    passage: str  # the r4 passage the extraction saw
    claims: list[SheetClaim]  # purposes then risks, in the row's order
