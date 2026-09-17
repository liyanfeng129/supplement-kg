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
