"""What the cleaning service hands out.

The shapes live beside the service that produces them, not in a global
`schemas/` package: a shape and the code that fills it drift apart the moment
they are edited in different files for different reasons.

These are deliberately *not* `repositories/dsld/labels/schema.py`'s `Label`. That one
is the frozen pilot row the gold set was labelled against, and its passage must
never be re-rendered; this one is the freshly cleaned reading of a raw record.
Two distinct types mean feeding a re-cleaned passage where a frozen one is
expected fails loudly instead of quietly shifting every score.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Segment:
    """One surviving piece of label text, with every place it came from.

    `paths` is a list rather than a single string because identical text
    routinely appears under several statements -- deduplication keeps one
    segment and collects the locations, so provenance survives the merge.
    Paths index the *raw* DSLD JSON (`statements[2].notes`), which is the
    document the audits and `gold_evidence_paths` point into.
    """

    text: str
    paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "paths": list(self.paths)}


@dataclass(frozen=True)
class CleanedLabel:
    """One raw DSLD record reduced to the text an extractor should see.

    `name` is carried for provenance and *also* appears as the first segment,
    under the path `fullName`. It is label text, and a product called "Sleep
    Aid" states its purpose there and sometimes nowhere else; the duplication
    is deliberate, so a caller can read the name without parsing the passage.
    """

    dsld_id: int
    name: str
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dsld_id": self.dsld_id,
            "name": self.name,
            "segments": [s.to_dict() for s in self.segments],
        }
