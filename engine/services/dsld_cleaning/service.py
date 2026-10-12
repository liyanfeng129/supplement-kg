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
name-sourced purpose can be filtered, weighted or audited downstream.

**No statement type is dropped.** Type names are not evidence of content:
gold-1000 raw-source review found a consultation warning under "General:
Product/Version Code", consultation advice mixed into an "FDA Disclaimer
Statement", weight-loss and respiratory claims under "Seals/Symbols", and
silver-1000 holds two purpose quotations available *only* under "FDA Statement
of Identity". The type itself is dropped from the text while surviving in
every segment's path. See docs/dsld-gold1000-field-audit.md and
docs/dsld-silver1000-field-audit.md for scope and limits.

**The text is normalised, and boilerplate is cut here** (renderer v4, spec
0005; this reverses v1-v3, which kept every segment as the label wrote it).
Every segment is lowercased, has its quantities removed (`500 mg` and
`1,000 mg` are the same label; names like `b12` and `omega-3` and age limits
like `under 18` are kept) and its punctuation turned to spaces, so copies of
one label stop looking different. Then every phrase on the checklist,
`boilerplate.txt`, is cut from every segment but `fullName`: the FDA
disclaimer alone is on 142,073 of 214,780 labels, and on a short label such
lines are most of the text, which makes unrelated products look alike and
was 33 of 38 false-positive risk claims on gold-100.

Only the phrase is cut, never the segment, so a mixed line keeps its real
caution ("keep out of reach of children discontinue use if rapid heartbeat
occurs" keeps the second half). Grounding still holds: the model reads the
v4 passage and its quotes are checked against that same passage. Only raw
DSLD is immutable; label sets saved under v1-v3 stay as they were written.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from datasketch import MinHash, MinHashLSH

from engine.repositories.dsld.labels import store
from engine.services.dsld_cleaning.schema import CleanedLabel, Cluster, Segment

# Hand-bumped, and stamped onto every flattened row. 1 = the renderer the pilot
# and gold set were built with: three skipped statement types, a 25-ingredient
# cap, `[type] notes` lines. 2 = text-only statements across all types plus
# recursive ingredient free text; silver-1000 was drawn against this. 3 = adds
# `fullName` as the first segment. 4 = normalised text, quantities removed,
# checklist boilerplate cut (spec 0005); editing boilerplate.txt bumps it too.
#
# It rides on the row because a v3 passage file is otherwise indistinguishable
# from a v2 one on disk, and gold was labelled against v1 and silver against
# v2 -- scoring the wrong pair together shifts every number silently.
RENDERER_VERSION = 4

# A quantity: a number (`1,000`, `2.5`), optionally a range (`2-3`), then
# either a unit, with or without a space, or nothing at all -- a standalone
# number. A number touching a letter or a hyphen on either side is a name
# (`b12`, `omega-3`, `5-htp`), so neither side may touch one. Age limits are
# kept by `_normalise` itself, which needs the words around the match.
_UNITS = (r"fl\s*oz|mcg|µg|μg|ug|mg|kg|g|iu|ml|l|oz|lbs|lb|kcal|cal|%|cfu|billion|million"
          r"|capsules?|caps|tablets?|softgels?|vegicaps?|gummy|gummies|drops?|scoops?"
          r"|packets?|servings?|teaspoons?|tsp|tablespoons?|tbsp")
_NUMBER = r"\d+(?:[.,]\d+)*"
_QUANTITY = re.compile(
    rf"(?<![a-z0-9-]){_NUMBER}(?:\s*[-–]\s*{_NUMBER})?"
    rf"(?:\s*(?:{_UNITS})(?![a-z0-9])|(?![a-z0-9-]))"
)
_AGE_BEFORE = re.compile(r"\b(?:under|over|below|above|aged|age|ages|age\s+of)\s*$")
_AGE_AFTER = re.compile(r"\s*(?:years?|yrs?|months?)\b")
_PUNCTUATION = re.compile(r"[^\w\s]|_")

# MinHash's seed is fixed so the same rows always give the same clusters.
_MINHASH_SEED = 1

# The checklist, read once. Filled in below the class, because each phrase is
# run through `_normalise` first so it matches what the passages have become.
_BOILERPLATE_PATH = Path(__file__).with_name("boilerplate.txt")


class DsldCleaningService:
    """Cleans raw DSLD records into extractor-ready text.

    Holds no state. `clean`, `flatten` and `cluster` do no I/O: raw records
    come from the repository's loader, so they stay a pure transformation that
    a test can exercise without a 215k-file dump on disk. `save` hands the
    flattened rows to the store as a label set, which is what a run reads.
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

    def cluster(self, flattened: list[dict], threshold: float = 0.9,
                shingle: int = 5, num_perm: int = 128) -> list[Cluster]:
        """Group near-duplicate labels by MinHash LSH over `passage` (spec 0006).

        Clusters are stars, not connected components: each row joins the most
        similar existing representative whose estimated Jaccard is at least
        `threshold` (ties go to the earliest), or becomes a representative
        itself. Joining every similar pair instead would chain A~B~C into one
        cluster even when A and C are not alike. The representative is the
        first row of its cluster in input order, so the caller picks it by
        ordering the rows. Reads only `dsld_id`, `passage` and
        `renderer_version`; changes no row.
        """
        # Validate: unique ids, one renderer version of at least 4 (an older
        # passage still has case, amounts and boilerplate, and two versions
        # are normalised two ways), parameters in range.
        ids = [r["dsld_id"] for r in flattened]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate dsld_id in the rows to cluster")
        versions = {r.get("renderer_version") for r in flattened}
        if None in versions or any(v < 4 for v in versions):
            raise ValueError(f"clustering needs renderer v4 or later passages, got {versions}")
        if len(versions) > 1:
            raise ValueError(f"rows come from more than one renderer version: {versions}")
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if shingle < 1 or num_perm < 1:
            raise ValueError(f"shingle and num_perm must be at least 1, got {shingle}, {num_perm}")

        # The LSH only proposes candidates; it is built below the threshold so
        # pairs just above it still collide, and the join check drops the rest.
        lsh = MinHashLSH(threshold=max(threshold - 0.1, 0.5), num_perm=num_perm)
        members: dict[int, list[int]] = {}       # representative -> members, in first-appearance order
        position: dict[int, int] = {}            # representative -> its cluster_id
        signatures: dict[int, MinHash] = {}      # representative -> its signature
        cluster_of: dict[str, int] = {}          # passage seen so far -> its cluster's representative

        for r in flattened:
            dsld_id, passage = r["dsld_id"], r["passage"]

            # An exact repeat goes with its twin, with no signature or query.
            # This also gathers every empty passage behind the first one,
            # which is never put in the LSH, so nothing else can join it.
            if passage in cluster_of:
                members[cluster_of[passage]].append(dsld_id)
                continue

            best = None
            if passage:
                signature = self._signature(self._shingles(passage, shingle), num_perm)
                best_similarity = threshold
                for candidate in sorted(lsh.query(signature), key=position.__getitem__):
                    similarity = signature.jaccard(signatures[candidate])
                    if similarity > best_similarity or (best is None and similarity == best_similarity):
                        best, best_similarity = candidate, similarity

            if best is not None:
                members[best].append(dsld_id)
                cluster_of[passage] = best
            else:
                members[dsld_id] = [dsld_id]
                position[dsld_id] = len(position)
                cluster_of[passage] = dsld_id
                if passage:
                    signatures[dsld_id] = signature
                    lsh.insert(dsld_id, signature)

        return [Cluster(cluster_id=position[rep], representative=rep, members=m)
                for rep, m in members.items()]

    # --- cleaning steps -----------------------------------------------------

    def _clean_one(self, record: dict) -> CleanedLabel:
        """Apply every step to one record: select, normalise, strip, drop, deduplicate.
            sum of two list is concatenation of two lists,
            so the statement and ingredient segments are combined into one list of tuples.
        """

        found = (self._name_segments(record)
                 + self._statement_segments(record)
                 + self._ingredient_segments(record))

        # Normalise every segment; cut boilerplate from all but the name, which
        # is the label's own headline and never boilerplate; drop what is empty.
        kept = []
        for text, path in found:
            text = self._normalise(text)
            if path != "fullName":
                text = self._strip_boilerplate(text)
            if text:
                kept.append((text, path))

        # `name` stays raw: it is provenance, not model input.
        return CleanedLabel(
            dsld_id=int(record["id"]),
            name=str(record.get("fullName") or ""),
            segments=self._deduplicate(kept),
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
    def _normalise(text: str) -> str:
        """Lowercase, remove quantities, punctuation to spaces, collapse whitespace.

        Quantities go before punctuation, so `1,000` and `2.5` are still one
        number when they are removed. A number right after `under`, `age of`
        and the like, or right before `years`/`months`, is an age limit -- a
        caution, not an amount -- and is kept.
        """
        text = text.lower()

        def drop_quantity(match: re.Match) -> str:
            if (_AGE_BEFORE.search(text, 0, match.start())
                    or _AGE_AFTER.match(text, match.end())):
                return match.group(0)
            return " "

        text = _QUANTITY.sub(drop_quantity, text)
        text = _PUNCTUATION.sub(" ", text)
        return " ".join(text.split())

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        """Cut every checklist phrase from normalised text; collapse what is left.

        Phrases match on word boundaries and are tried longest first, so
        `warning keep out of reach of children` goes whole rather than leaving
        `warning` behind. Only the phrase goes; the rest of the segment stays.
        """
        return " ".join(_BOILERPLATE.sub(" ", text).split())

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

    # --- clustering ---------------------------------------------------------

    @staticmethod
    def _shingles(text: str, k: int) -> set[str]:
        """Every character `k`-gram of the text; a text shorter than `k` is one shingle."""
        if len(text) < k:
            return {text}
        return {text[i:i + k] for i in range(len(text) - k + 1)}

    @staticmethod
    def _signature(shingles: set[str], num_perm: int) -> MinHash:
        """MinHash the shingles with the fixed seed, all in one batch."""
        signature = MinHash(num_perm=num_perm, seed=_MINHASH_SEED)
        signature.update_batch([s.encode("utf-8") for s in shingles])
        return signature


# Load the checklist: skip blanks and comments, normalise each phrase so one
# written with a number or punctuation still matches, drop duplicates and
# phrases left empty, and compile longest first on word boundaries.
_phrases = {
    DsldCleaningService._normalise(line)
    for line in _BOILERPLATE_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}
_phrases.discard("")
_BOILERPLATE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(p) for p in sorted(_phrases, key=len, reverse=True))
    + r")(?!\w)"
)


# Module-level entries, so a caller that wants the default behaviour need not
# know the service is a class.
_service = DsldCleaningService()


def clean(records):
    """Clean one raw DSLD record, or many. See `DsldCleaningService.clean`."""
    return _service.clean(records)


def flatten(cleaned):
    """Render cleaned labels into storable rows. See `DsldCleaningService.flatten`."""
    return _service.flatten(cleaned)


def cluster(flattened, threshold=0.9, shingle=5, num_perm=128):
    """Group near-duplicate labels. See `DsldCleaningService.cluster`."""
    return _service.cluster(flattened, threshold, shingle, num_perm)
