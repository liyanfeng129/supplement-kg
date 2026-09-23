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

# No type is safe to drop wholesale: gold-1000 raw-source review found
# substantive text even under disclaimer, identity and product-code types.
# Retained as an empty compatibility export for existing callers.
SKIP_TYPES: set[str] = set()

# The prompt text lives in `prompts/purpose_risk.md` and is read from there, not
# duplicated here. It is the one editable file per prompt step that ADR 0001 §4
# settled on, each run's manifest embeds it verbatim, and `config_hash` covers
# its sha -- so a second copy in this module would be a second source of truth
# that drifts silently: the local pipeline would hash the file while this
# `extract()` sent something else, and the gold set was built through here.
PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "purpose_risk.md"

SYSTEM = PROMPT_PATH.read_text().rstrip("\n")

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


def label_segments(label: Label) -> list[dict]:
    """Deduplicated extraction text with paths back to the raw DSLD record.

    Drop structured metadata and type/name headers from model input. Keep
    free-text statements across all types and ingredient notes/descriptions:
    zero gold matches does not certify an unannotated field as irrelevant.
    Paths refer to the raw JSON, with duplicate locations retained together.
    """
    segments: dict[str, dict] = {}

    def add(text: str | None, path: str) -> None:
        text = " ".join((text or "").split())
        if text:
            segment = segments.setdefault(text, {"text": text, "paths": []})
            segment["paths"].append(path)

    for index, statement in enumerate(label.statements):
        add(statement.get("notes"), f"statements[{index}].notes")

    def ingredients(rows, prefix):
        for index, ingredient in enumerate(rows):
            path = f"{prefix}[{index}]"
            add(ingredient.notes, f"{path}.notes")
            add(ingredient.description, f"{path}.description")
            ingredients(ingredient.nested, f"{path}.nestedRows")

    ingredients(label.ingredients, "ingredientRows")
    return list(segments.values())


def label_text(label: Label) -> str:
    """Render only extraction text; identity and source paths stay outside it.

    All gold-1000 quotations survive without metadata headers or statement
    type prefixes. See docs/dsld-gold1000-field-audit.md for scope and limits.
    """
    return "\n".join(segment["text"] for segment in label_segments(label))


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
