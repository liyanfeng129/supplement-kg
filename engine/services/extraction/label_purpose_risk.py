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

# Statement types worth sending to the model. DSLD's `type` vocabulary is
# coarse -- purpose routinely hides in the generic "General Statements" bucket
# -- so this filter is deliberately loose; it drops only the types that never
# carry purpose or risk (lot codes, brand trademark notices, seals).
INFORMATIVE_PREFIXES = (
    "General Statements",
    "Precautions",
    "Suggested",
    "Formula",
    "Formulation",
    "Storage",
    "FDA Statement of Identity",
    "Warnings",
)

SKIP_TYPES = {
    "General: Product/Version Code",
    "Brand IP Statement(s) re: (c), (TM), (SM)",
    "Seals/Symbols",
    "FDA Disclaimer Statement",
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
- Set a field to null when the label genuinely does not state it. Empty is a
  valid and common answer -- prefer it over a guess."""

# Structured output schema. `strict` validation guarantees the response parses,
# so downstream code never has to defend against malformed JSON.
SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {
            "type": ["string", "null"],
            "description": "Claimed benefit or intended use, in the label's own wording. Null if not stated.",
        },
        "risk": {
            "type": ["string", "null"],
            "description": "Cautions, warnings, contraindications, interactions. Null if not stated.",
        },
        "purpose_evidence": {
            "type": ["string", "null"],
            "description": "Verbatim label text the purpose was taken from. Null if purpose is null.",
        },
        "risk_evidence": {
            "type": ["string", "null"],
            "description": "Verbatim label text the risk was taken from. Null if risk is null.",
        },
    },
    "required": ["purpose", "risk", "purpose_evidence", "risk_evidence"],
    "additionalProperties": False,
}


@dataclass
class Extraction:
    """One product's recovered Purpose/Risk, with the evidence behind it."""

    dsld_id: int
    idisk_id: str | None = None
    name: str | None = None
    purpose: str | None = None
    risk: str | None = None
    # Verbatim label text each field came from. Keeping it makes a wrong
    # extraction auditable without re-querying the API -- and makes an
    # ungrounded answer (evidence absent from the label) detectable.
    purpose_evidence: str | None = None
    risk_evidence: str | None = None
    model: str = MODEL
    extracted_at: str | None = None


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
    for s in label.statements:
        stype, notes = (s.get("type") or ""), (s.get("notes") or "").strip()
        if not notes or stype in SKIP_TYPES:
            continue
        if not stype.startswith(INFORMATIVE_PREFIXES):
            continue
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
        messages=[{"role": "user", "content": label_text(label)}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "{}")
    data = json.loads(text)

    return Extraction(
        dsld_id=label.dsld_id,
        idisk_id=label.idisk_id,
        name=label.full_name,
        purpose=data.get("purpose"),
        risk=data.get("risk"),
        purpose_evidence=data.get("purpose_evidence"),
        risk_evidence=data.get("risk_evidence"),
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
