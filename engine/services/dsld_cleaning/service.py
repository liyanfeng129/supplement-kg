"""Reduce a raw DSLD record to the text an extractor should see.

**Selection is an allowlist.** Five paths are read out of the raw JSON:

    fullName
    statements[i].notes
    ingredientRows[i].notes
    ingredientRows[i].description
    ingredientRows[i].nestedRows[...]        (recursively, same three)

Everything else is absent because no line here reads it -- not because it is
deleted. A denylist naming `upcSku`, `events`, `labelRelationships` and the
rest would be thirty lines instead of four and, worse, would silently admit
any field DSLD adds later into the model input.

**What is left out, and why.** Operational metadata (`src`, `id`, `nhanesId`,
`bundleName`, `upcSku`, `pdf`, `thumbnail`, `labelRelationships`, `events`,
`entryDate`, `offMarket`, `hasOuterCarton`), contact and packaging details,
structured ingredient identity and dosage (ids, quantities, units, daily
values), and the `claims` / `productType` / `physicalState` / `targetGroups` /
brand and ingredient-name headers. All are preserved in the raw record for
provenance and for ingredient and dosage KG work.

**`fullName` is in the passage, and that is a reversal.** It was held out on
the grounds that grounding is a substring test and text the label never stated
as label text must not become quotable. The premise was wrong: a product name
is label text, and under FDA rules a structure/function claim in the name is a
claim. A product called "Sleep Aid" states its intended use in its name.

A 2,000-label audit of everything this service drops (23,964 distinct strings,
every one read) found purpose content in exactly five dropped paths, all of
them name fields, and `fullName` carried 19 of the 20 that plainly assert a
recordable claim -- "Sleep Aid", "Cholesterol Health", "Liver Cleanse
Formula", "Urinary Maintenance". Usually the kept statements say it too, so
the name is redundant; sometimes they do not, and then dropping the name loses
the only purpose the label states. DSLD 230595 ("Sleep Aid") is the case in
point: its statements are dosing, cautions and FDA boilerplate, and nothing in
them says what the product is for.

The name is quotable, deliberately: if the name *is* the claim, a model has to
be able to cite it. What keeps that honest is the path -- a claim grounded in
`fullName` is distinguishable from one grounded in `statements[3].notes`, so
name-sourced purpose can be filtered, weighted or audited downstream. That is
the same output-side filtering this module already insists on for boilerplate.

**No statement type is dropped.** Type names are not evidence of content:
gold-1000 raw-source review found a consultation warning under "General:
Product/Version Code", consultation advice mixed into an "FDA Disclaimer
Statement", weight-loss and respiratory claims under "Seals/Symbols", and
silver-1000 holds two purpose quotations available *only* under "FDA Statement
of Identity". The type itself is dropped from the text while surviving in
every segment's path. See docs/dsld-gold1000-field-audit.md and
docs/dsld-silver1000-field-audit.md for scope and limits.

**No boilerplate filtering happens here.** A passage that lost a line can
never get it back, while an output filter is reversible and testable, so
boilerplate suppression belongs on the extraction output -- a mixed statement
carrying both a lot code and a caution must survive this stage intact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from engine.repositories.dsld.labels import store
from engine.services.dsld_cleaning.schema import CleanedLabel, Segment

# Hand-bumped, and stamped onto every flattened row. 1 = the renderer the pilot
# and gold set were built with: three skipped statement types, a 25-ingredient
# cap, `[type] notes` lines. 2 = text-only statements across all types plus
# recursive ingredient free text; silver-1000 was drawn against this. 3 = adds
# `fullName` as the first segment.
#
# It rides on the row because a v3 passage file is otherwise indistinguishable
# from a v2 one on disk, and gold was labelled against v1 and silver against
# v2 -- scoring the wrong pair together shifts every number silently.
RENDERER_VERSION = 3


class DsldCleaningService:
    """Cleans raw DSLD records into extractor-ready text.

    Holds no state. `clean` and `flatten` do no I/O: raw records come from
    the repository's loader, so they stay a pure transformation that a test
    can exercise without a 215k-file dump on disk. `save` hands the flattened
    rows to the store as a label set, which is what a run reads.
    """

    # --- public -------------------------------------------------------------

    def clean(self, records):
        """Clean one raw DSLD record, or many.

        A single record returns a single `CleanedLabel`; an iterable returns a
        list, matching the loader's own one-or-many convention so a caller can
        pass `load_raw(...)` straight through whatever quantity it asked for.
        """
        if isinstance(records, dict):
            return self._clean_one(records)
        return [self._clean_one(r) for r in records]

    def flatten(self, cleaned):
        """Render cleaned labels into rows ready for a model and a store.

        `passage` is the whole label text as one string -- what a prompt sends
        and what grounding is checked against. `segments` rides along as
        provenance rather than being exploded into its own rows: the paths
        describe the row, they are not rows themselves.
        """
        if isinstance(cleaned, CleanedLabel):
            return self._flatten_one(cleaned)
        return [self._flatten_one(c) for c in cleaned]

    def save(self, flattened: list[dict], name: str) -> str:
        """Save flattened rows as label set `name`; return its `passages_sha`.

        A name is one set of passages for good: saving the same rows again
        writes nothing, and saving different ones under a used name is
        refused. Name a set by what distinguishes it -- `silver1000-r2` is
        1000 silver labels, renderer 2.
        """
        return store.write_label_set(
            name,
            flattened,
            {
                "source": "engine.services.dsld_cleaning",
                "renderer_version": RENDERER_VERSION,
                "n_labels": len(flattened),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # --- cleaning steps -----------------------------------------------------

    def _clean_one(self, record: dict) -> CleanedLabel:
        """Apply every step to one record: select, collapse, deduplicate.
            sum of two list is concatenation of two lists, 
            so the statement and ingredient segments are combined into one list of tuples.
        """
        
        found = (self._name_segments(record)
                 + self._statement_segments(record)
                 + self._ingredient_segments(record))
        return CleanedLabel(
            dsld_id=int(record["id"]),
            name=str(record.get("fullName") or ""),
            segments=self._deduplicate(found),
        )

    def _name_segments(self, record: dict) -> list[tuple[str, str]]:
        """Read the product name, which often states the product's purpose.

        First in the passage because it is the label's own headline, and a
        model reading the statements afterwards should already know what the
        product calls itself. The path is `fullName`, so a purpose grounded in
        the name is always separable from one grounded in a statement.
        """
        text = self._collapse_whitespace(record.get("fullName"))
        return [(text, "fullName")] if text else []

    def _statement_segments(self, record: dict) -> list[tuple[str, str]]:
        """Read `notes` off every statement, whatever its type.

        The type is not read into the text and not used to filter; it survives
        only inside the path.
        """
        out = []
        for index, statement in enumerate(record.get("statements") or []):
            if not isinstance(statement, dict):
                continue
            text = self._collapse_whitespace(statement.get("notes"))
            if text:
                out.append((text, f"statements[{index}].notes"))
        return out

    def _ingredient_segments(self, record: dict) -> list[tuple[str, str]]:
        """Read ingredient free text, descending `nestedRows` to any depth.

        A proprietary blend nests its constituents, and benefit language sits
        at whatever level the label put it, so the walk cannot stop at the top
        row. Paths keep the full descent (`ingredientRows[0].nestedRows[2]...`)
        so a segment found six levels down still names where it came from.
        """
        out: list[tuple[str, str]] = []

        def walk(rows, prefix: str) -> None:
            for index, row in enumerate(rows or []):
                if not isinstance(row, dict):
                    continue
                path = f"{prefix}[{index}]"
                for field_name in ("notes", "description"):
                    text = self._collapse_whitespace(row.get(field_name))
                    if text:
                        out.append((text, f"{path}.{field_name}"))
                walk(row.get("nestedRows"), f"{path}.nestedRows")

        walk(record.get("ingredientRows"), "ingredientRows")
        return out

    @staticmethod
    def _collapse_whitespace(text) -> str:
        """Squeeze runs of whitespace to single spaces; empty stays empty.

        Line breaks inside a DSLD note are layout, not meaning, and leaving
        them in would make a model's faithful quotation fail a substring check
        on formatting rather than on fabrication.
        """
        return " ".join(str(text or "").split())

    @staticmethod
    def _deduplicate(found: Iterable[tuple[str, str]]) -> list[Segment]:
        """Merge identical text into one segment, keeping every path.

        The same sentence under three statements is one thing the label says,
        and sending it three times pays for it three times. First-seen order is
        preserved, so statements precede ingredients in the passage.
        """
        merged: dict[str, list[str]] = {}
        for text, path in found:
            merged.setdefault(text, []).append(path)
        return [Segment(text=t, paths=p) for t, p in merged.items()]

    # --- flattening ---------------------------------------------------------

    @staticmethod
    def _flatten_one(cleaned: CleanedLabel) -> dict:
        return {
            "dsld_id": cleaned.dsld_id,
            "name": cleaned.name,
            "passage": "\n".join(s.text for s in cleaned.segments),
            "segments": [s.to_dict() for s in cleaned.segments],
            "renderer_version": RENDERER_VERSION,
        }


# Module-level entries, so a caller that wants the default behaviour need not
# know the service is a class.
_service = DsldCleaningService()


def clean(records):
    """Clean one raw DSLD record, or many. See `DsldCleaningService.clean`."""
    return _service.clean(records)


def flatten(cleaned):
    """Render cleaned labels into storable rows. See `DsldCleaningService.flatten`."""
    return _service.flatten(cleaned)
