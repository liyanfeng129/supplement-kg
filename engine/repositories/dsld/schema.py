"""What the DSLD repository hands out and takes back.

The shapes live beside the loader and store that produce them, not in a global
`schemas/` package: a shape and the code that fills it drift apart the moment
they are edited in different files for different reasons.

`Label` is deliberately not the DSLD API's own record. It is the *frozen*
shape the pilot wrote and the gold set was built against -- id, name and one
rendered passage. Re-rendering a label from the raw dump would produce text the
gold set never saw and quietly break every comparison against it, so the
passage is read as data rather than rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    """One label as an extractor sees it: an id, a name, and the passage.

    Frozen because a label is input. A service that wants to change what the
    model sees changes the passage builder, not the record it was handed --
    mutating this in place would leave the run's manifest describing text that
    no longer exists.
    """

    dsld_id: int
    name: str
    passage: str

    @classmethod
    def from_dict(cls, record: dict) -> "Label":
        """Build from a stored JSONL record, failing loudly on a bad one.

        `dsld_id` is coerced to int because it is written as a string in some
        upstream DSLD exports and as a number in the pilot; a set of mixed ids
        silently fails every membership test that resume depends on.
        """
        try:
            return cls(
                dsld_id=int(record["dsld_id"]),
                name=str(record.get("name") or ""),
                passage=str(record["passage"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a usable label record: {record!r}") from exc

    def to_dict(self) -> dict:
        return {"dsld_id": self.dsld_id, "name": self.name, "passage": self.passage}


@dataclass(frozen=True)
class Claim:
    """One extracted purpose or risk, with the span it was read from.

    `evidence` is what makes a gold row auditable: the gold set is one model's
    reading of a label, not a human adjudication, so a claim that cannot be
    pointed back at the passage cannot be checked. `grounded` records whether
    that span was found in the passage verbatim at write time.
    """

    text: str
    evidence: str = ""
    grounded: bool | None = None

    @classmethod
    def from_dict(cls, record: dict) -> "Claim":
        try:
            return cls(
                text=str(record["text"]),
                evidence=str(record.get("evidence") or ""),
                grounded=record.get("grounded"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"not a usable claim: {record!r}") from exc

    def to_dict(self) -> dict:
        return {"text": self.text, "evidence": self.evidence, "grounded": self.grounded}


@dataclass(frozen=True)
class GoldLabel:
    """The reference reading of one label: its purposes and its risks.

    A *reference*, not ground truth -- see `services/extraction/gold_set.py`.
    It carries `model` so a later comparison can say which reading it scored
    against, and the 100 rows are committed outside the gitignored `data/`
    tree because regenerating them costs money.
    """

    dsld_id: int
    name: str = ""
    purpose: list[Claim] = None  # type: ignore[assignment]
    risk: list[Claim] = None  # type: ignore[assignment]
    model: str = ""
    extracted_at: str = ""

    def __post_init__(self) -> None:
        # Frozen, so the empty-list defaults are filled through object.__setattr__
        # rather than a mutable default shared by every instance.
        if self.purpose is None:
            object.__setattr__(self, "purpose", [])
        if self.risk is None:
            object.__setattr__(self, "risk", [])

    @classmethod
    def from_dict(cls, record: dict) -> "GoldLabel":
        try:
            return cls(
                dsld_id=int(record["dsld_id"]),
                name=str(record.get("name") or ""),
                purpose=[Claim.from_dict(c) for c in record.get("purpose") or []],
                risk=[Claim.from_dict(c) for c in record.get("risk") or []],
                model=str(record.get("model") or ""),
                extracted_at=str(record.get("extracted_at") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a usable gold record: {record!r}") from exc

    def to_dict(self) -> dict:
        return {
            "dsld_id": self.dsld_id,
            "name": self.name,
            "purpose": [c.to_dict() for c in self.purpose],
            "risk": [c.to_dict() for c in self.risk],
            "model": self.model,
            "extracted_at": self.extracted_at,
        }
