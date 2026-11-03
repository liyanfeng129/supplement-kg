"""What the runs repository hands out and takes back beyond a run's raw rows.

A run's rows stay plain dicts, as the `Sink` wrote them. The profile of a
finished run -- each label's group and the type of each of its claims -- is
stored beside them, and its shape lives here, with the store and loader that
write and read it, not with the service that computes it (ADR 0002 §5). So
does the evaluation's sample of a run (spec 0011.02): the labels drawn, their
strata, and the claims a read joins to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["purpose", "risk"]
Type = Literal["verbatim", "stitched", "spread", "invented"]
Group = Literal["both", "purpose_only", "risk_only", "neither"]
Rarest = Literal["rare", "stitched", "verbatim"]
Stratum = tuple[Group, Rarest | None]

# The 10 strata, in the fixed order every table and manifest uses: each group
# with claims × each rarest type, then `neither`, which has no claims.
STRATA: list[Stratum] = [
    *((g, r) for g in ("both", "purpose_only", "risk_only") for r in ("rare", "stitched", "verbatim")),
    ("neither", None),
]


@dataclass(frozen=True)
class ProfiledLabel:
    """One label of a run: its group, and the type of each of its claims.

    No claim ids and no claim text: `purpose_types[i]` is the type of
    `claim:<dsld_id>:purpose:<i>`, whose text is `row["purpose"][i]["text"]`.
    The index is stable because the profile's manifest pins the rows.
    """

    dsld_id: int
    group: Group
    purpose_types: list[Type]
    risk_types: list[Type]


@dataclass(frozen=True)
class SampledLabel:
    """One label drawn for the evaluation, stored as ids and its stratum only."""

    case_id: str  # label:<dsld_id>
    dsld_id: int
    stratum: Stratum
    rank: int  # position in the stratum's seeded order, 0-based
    in_human: bool  # pilot: the first pilot_human; large set: the first human


@dataclass(frozen=True)
class StratumSize:
    """How many labels of one stratum are in the run and in each set."""

    stratum: Stratum
    in_run: int  # N_s
    pilot: int  # p_s
    pilot_human: int  # q_s
    large: int  # n_s, after "all" is applied
    human: int  # k_s


@dataclass(frozen=True)
class Claim:
    """One claim of a sampled label: its text from the run's row, its type from the profile."""

    claim_id: str  # claim:<dsld_id>:<kind>:<index>
    text: str
    type: Type


@dataclass(frozen=True)
class SampledCase:
    """A sampled label joined with its claims, as a read of the pilot or evaluation set returns it."""

    label: SampledLabel
    purpose: list[Claim]  # in the row's order: purpose[i] is claim:<dsld_id>:purpose:<i>
    risk: list[Claim]
