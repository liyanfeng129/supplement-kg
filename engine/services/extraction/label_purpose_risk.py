"""LLM extraction of product purpose and risk from DSLD label text.

iDISK 2.0 sources Purpose and Risk from LNHPD only, so all 92,442 DSLD-only
DSP rows have both fields empty (see notebooks/DSP_processing.ipynb). DSLD
does carry the underlying information, but as unlabelled free text: the
purpose of a sleep aid may sit in a `General Statements: All Other Content`
block between "Made in the USA" and "ACTUAL SIZE". This module recovers the
two fields from that text so DSLD-only products can carry the same attributes
LNHPD-sourced ones do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import anthropic

from engine import config
from engine.services.sources.dsld import Label

MODEL = "claude-opus-5"

# Statement types dropped before the label is sent to a model. Only these
# three are verified safe across all 300 sampled labels: they are fixed
# boilerplate (the DSHEA disclaimer, "Dietary Supplement", lot codes) and
# carried zero benefit or risk language.
#
# Nothing else is dropped. An earlier version also dropped `Brand IP
# Statement(s) re: (c), (TM), (SM)` and `Seals/Symbols`, plus allow-listed a
# set of INFORMATIVE_PREFIXES. Both filters were wrong: Brand IP carries real
# benefit claims in 21 of 300 sampled statements and is the *only* purpose
# text on labels 1597, 38206 and 176329. DSLD's `type` vocabulary is too
# coarse to filter on -- see docs/dsld-purpose-risk-findings.md section 5.
SKIP_TYPES = {
    "FDA Statement of Identity",
    "FDA Disclaimer Statement",
    "General: Product/Version Code",
}

SYSTEM = """You extract structured attributes from US dietary supplement label text.

You are given the text printed on one product's label. Recover two fields:

- purpose: what the product is claimed to do for the person taking it -- the
  benefit or intended use ("supports restful sleep", "helps maintain healthy
  joints"). This is the product's structure/function claim.
- risk: cautions, warnings, contraindications, and interactions -- who should
  not take it and what to watch for.

Rules:
- Use ONLY what the label says. Never add pharmacological knowledge of your
  own: if a label for a melatonin product never states a purpose, the purpose
  is null, even though you know what melatonin is for. This corpus is used to
  build a knowledge base, so an inferred claim is worse than a missing one.
- Quote or lightly normalise the label's own wording; do not embellish.
- Purpose statements are often buried in generic "General Statements" blocks
  alongside boilerplate ("Made in the USA", "ACTUAL SIZE"). Ignore the
  boilerplate and pick out the benefit claim.
- The FDA disclaimer ("This statement has not been evaluated by the FDA...")
  is not a purpose. Directions/dosage alone are not a purpose.
- "KEEP OUT OF REACH OF CHILDREN" alone is boilerplate, not a risk; include it
  only if no other risk text exists.
- Return an empty list when the label genuinely does not state a field. Empty
  is a valid and common answer -- prefer it over a guess.

Output shape:
- purpose and risk are LISTS, one entry per distinct claim. A label stating
  "Promotes Energy Metabolism*" and "Supports Heart Health*" yields two
  purpose entries, not one joined string.
- Each entry has `evidence` -- characters copied from the label text above,
  unchanged, so the claim can be checked against the label -- and `text`, the
  same claim normalised: lowercased, trailing claim asterisk removed. Never
  write anything into `evidence` that is not in the text above.
- Where several distinct cautions share one block of label text and no
  substring isolates one of them, cite the block as `evidence` for each and
  put the individual caution in `text`.
- purpose_confidence and risk_confidence score 0-5 how clearly THE LABEL
  states that field -- 0 not at all, 5 explicitly and unambiguously. They
  score the clarity of the extraction, not how risky or effective the product
  is. Never judge a product\'s safety."""

# Structured output schema. `strict` validation guarantees the response parses,
# so downstream code never has to defend against malformed JSON.
#
# Purpose and risk are lists, one entry per distinct claim, not single
# strings. A label routinely states several benefits, and DSLD packs real
# cautions into one `notes` block together with boilerplate ("keep out of
# reach of children", "store in a cool, dry place"). A scalar field would
# land that whole block in `Risk` verbatim -- the dilution problem in section
# 9 of the findings doc, and unfixable afterwards. Per-entry rows give a
# boilerplate-suppression pass somewhere to act. Collapsing a list into one
# string later is trivial; splitting a joined string back apart is not.
_ENTRY = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "The single claim, normalised: lowercased, trailing claim asterisk removed.",
        },
        "evidence": {
            "type": "string",
            "description": "Characters copied from the label text above, unchanged, that this claim came from.",
        },
    },
    "required": ["text", "evidence"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {
            "type": "array",
            "items": _ENTRY,
            "description": "One entry per distinct benefit or intended-use claim. Empty if the label states none.",
        },
        "risk": {
            "type": "array",
            "items": _ENTRY,
            "description": "One entry per distinct caution, contraindication or interaction. Empty if the label states none.",
        },
        "purpose_confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "description": "How clearly the label states its purpose: 0 not at all, 5 explicitly.",
        },
        "risk_confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 5,
            "description": "How clearly the label states its risks: 0 not at all, 5 explicitly.",
        },
    },
    "required": ["purpose", "risk", "purpose_confidence", "risk_confidence"],
    "additionalProperties": False,
}

# Ceiling applied to a self-reported confidence when an entry's evidence is
# not found in the passage. The model scores its own certainty, which a
# confident hallucination scores just as highly as a correct reading; the
# substring check cannot be talked up, so it bounds the score it cannot set.
UNGROUNDED_CONFIDENCE_CAP = 1


@dataclass
class Entry:
    """One extracted claim and the label text it came from."""

    text: str
    evidence: str
    # Whether `evidence` occurs verbatim in the passage. False means the model
    # produced text the label does not contain -- a fabrication, caught for
    # free without a second model or a human read.
    grounded: bool = True


@dataclass
class Extraction:
    """One product's recovered Purpose/Risk, with the evidence behind it."""

    dsld_id: int
    idisk_id: str | None = None
    name: str | None = None
    purpose: list[Entry] = field(default_factory=list)
    risk: list[Entry] = field(default_factory=list)
    # Scored per field, not per row: labels very often state a purpose
    # clearly and no risk at all, so one number would describe neither.
    purpose_confidence: int = 0
    risk_confidence: int = 0
    model: str = MODEL
    extracted_at: str | None = None


def _normalise(text: str) -> str:
    """Whitespace-insensitive form, for comparing evidence against a passage.

    A model reproducing a span across a line break in the passage is quoting
    faithfully; failing it for the whitespace would make the grounding check
    fire on formatting rather than on fabrication.
    """
    return " ".join(text.split()).lower()


def _entries(items: list[dict], passage: str) -> list[Entry]:
    """Build entries, marking each by whether its evidence is in the passage."""
    hay = _normalise(passage)
    out = []
    for it in items or []:
        ev = (it.get("evidence") or "").strip()
        out.append(
            Entry(
                text=(it.get("text") or "").strip(),
                evidence=ev,
                grounded=bool(ev) and _normalise(ev) in hay,
            )
        )
    return out


def _capped(confidence: int, entries: list[Entry]) -> int:
    """Bound a self-reported score by the mechanical grounding check."""
    if entries and not all(e.grounded for e in entries):
        return min(confidence, UNGROUNDED_CONFIDENCE_CAP)
    # Nothing extracted cannot be a confident answer, whatever was claimed.
    if not entries:
        return 0
    return confidence


def label_text(label: Label) -> str:
    """Render the parts of a label an extractor should read.

    Ingredients are included because a purpose is sometimes only inferable
    alongside them, and the claim-type list because "Structure/Function" tells
    the model a benefit claim is present somewhere on the label.
    """
    lines = [f"PRODUCT: {label.full_name}"]
    if label.brand_name:
        lines.append(f"BRAND: {label.brand_name}")
    if label.product_type:
        lines.append(f"TYPE: {label.product_type} ({label.physical_state})")
    if label.claims:
        lines.append(f"CLAIM TYPES ON LABEL: {', '.join(label.claims)}")
    if label.target_groups:
        lines.append(f"TARGET GROUPS: {', '.join(label.target_groups)}")

    ings = [i.name for i in label.ingredients if i.name][:25]
    if ings:
        lines.append(f"INGREDIENTS: {', '.join(ings)}")

    lines.append("\nLABEL STATEMENTS:")
    # The `[type]` prefix is kept: it is unreliable as a filter but useful as a
    # hint to a small model that a line came from `Precautions` rather than
    # `General Statements`.
    seen: set[str] = set()
    for s in label.statements:
        stype, notes = (s.get("type") or ""), (s.get("notes") or "")
        notes = " ".join(notes.split())
        if not notes or stype in SKIP_TYPES:
            continue
        # DSLD files the same text under several types on 2.4% of the corpus.
        if notes in seen:
            continue
        seen.add(notes)
        lines.append(f"[{stype}] {notes}")

    return "\n".join(lines)


def _client(api_key: str | None = None) -> anthropic.Anthropic:
    key = api_key or config.ANTHROPIC_API_KEY
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env "
            "(see .env.example) to run LLM extraction."
        )
    return anthropic.Anthropic(api_key=key)


def extract(
    label: Label,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> Extraction:
    """Recover purpose and risk for one label."""
    client = client or _client()
    passage = label_text(label)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM,
        # Effort is low deliberately: this is short-text extraction, not
        # reasoning, and the run is per-product across a large corpus.
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        messages=[{"role": "user", "content": passage}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "{}")
    data = json.loads(text)

    purpose = _entries(data.get("purpose"), passage)
    risk = _entries(data.get("risk"), passage)

    return Extraction(
        dsld_id=label.dsld_id,
        idisk_id=label.idisk_id,
        name=label.full_name,
        purpose=purpose,
        risk=risk,
        purpose_confidence=_capped(data.get("purpose_confidence") or 0, purpose),
        risk_confidence=_capped(data.get("risk_confidence") or 0, risk),
        model=model,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def extract_many(
    labels: Sequence[Label],
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
    on_error: str = "skip",
) -> tuple[list[Extraction], dict]:
    """Extract over many labels. Returns (extractions, manifest)."""
    client = client or _client()
    started = datetime.now(timezone.utc).isoformat()

    out: list[Extraction] = []
    errors: list[dict] = []
    for lb in labels:
        try:
            out.append(extract(lb, client=client, model=model))
        except Exception as exc:  # noqa: BLE001 - one bad label must not sink a run
            if on_error != "skip":
                raise
            errors.append({"dsld_id": lb.dsld_id, "error": repr(exc)})

    manifest = {
        "task": "dsld_purpose_risk",
        "model": model,
        "executed_at": started,
        "n_labels": len(labels),
        "n_extracted": len(out),
        "n_with_purpose": sum(1 for e in out if e.purpose),
        "n_with_risk": sum(1 for e in out if e.risk),
        "n_purpose_entries": sum(len(e.purpose) for e in out),
        "n_risk_entries": sum(len(e.risk) for e in out),
        # Ungrounded entries are the fabrication rate: evidence the model
        # produced that the label does not contain. Surfaced in the manifest
        # because it is the number that says whether a run is trustworthy.
        "n_ungrounded_entries": sum(
            1 for e in out for x in (*e.purpose, *e.risk) if not x.grounded
        ),
        "n_rows_with_ungrounded": sum(
            1 for e in out if any(not x.grounded for x in (*e.purpose, *e.risk))
        ),
        "errors": errors,
    }
    return out, manifest


def save_extractions(
    extractions: list[Extraction],
    manifest: dict,
    out_dir: str | Path,
    run_name: str = "purpose_risk",
) -> dict[str, Path]:
    """Write payload (gitignored) and manifest (committable) side by side."""
    out_dir = Path(out_dir)
    raw_dir, man_dir = out_dir / "raw", out_dir / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{run_name}.jsonl"
    with raw_path.open("w") as fh:
        for e in extractions:
            fh.write(json.dumps(asdict(e)) + "\n")

    man_path = man_dir / f"{run_name}.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    return {"raw": raw_path, "manifest": man_path}
