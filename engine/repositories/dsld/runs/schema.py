"""What the runs repository hands out and takes back beyond a run's raw rows.

A run's rows stay plain dicts, as the `Sink` wrote them. The profile of a
finished run -- each label's group and the type of each of its claims -- is
stored beside them, and its shape lives here, with the store and loader that
write and read it, not with the service that computes it (ADR 0002 §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["purpose", "risk"]
Type = Literal["verbatim", "stitched", "spread", "invented"]
Group = Literal["both", "purpose_only", "risk_only", "neither"]


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
