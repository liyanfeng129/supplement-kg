"""Raw model output to one validated, reviewable row.

Every driver's rows pass through here, so the contract is enforced rather than
merely documented. Three things happen, in this order:

- **Grounding.** Each entry's `evidence` is checked for a verbatim (whitespace-
  insensitive) occurrence in the passage. False means the model produced text
  the label does not contain -- a fabrication, caught without a second model or
  a human read.
- **The confidence cap.** A self-reported score is bounded by that mechanical
  check, because a confident hallucination scores itself just as highly as a
  correct reading. An empty extraction scores 0: "found nothing" is never a
  confident answer.
- **Status.** `ok`, `empty`, `unparseable` or `error`. `empty` is a first-class
  success -- the gold set says no purpose is correct on 28% of labels and no
  risk on 29% -- and conflating it with `unparseable` would hide exactly what
  `smoke` exists to catch.

A GBNF grammar guarantees *shape*; this guarantees nothing about truthfulness
either, but it measures it. Both are kept. Raw text is retained on every row
whatever the status, so an unparseable result stays diagnosable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from engine.services.extraction.label_purpose_risk import (
    UNGROUNDED_CONFIDENCE_CAP,
    _normalise,
)

STATUSES = ("ok", "empty", "unparseable", "error", "poison")


def _entries(items: Any, passage: str) -> list[dict]:
    """Build entry dicts, marking each by whether its evidence is in the passage."""
    hay = _normalise(passage)
    out: list[dict] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        ev = str(it.get("evidence") or "").strip()
        out.append(
            {
                "text": str(it.get("text") or "").strip(),
                "evidence": ev,
                "grounded": bool(ev) and _normalise(ev) in hay,
            }
        )
    return out


def _confidence(value: Any, entries: list[dict]) -> int:
    """Bound a self-reported score by the mechanical grounding check."""
    # The grammar constrains this to an integer, but the agent and api drivers
    # do not go through the grammar, and a wrong type must not sink a row.
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 0

    if not entries:
        return 0
    if not all(e["grounded"] for e in entries):
        return min(score, UNGROUNDED_CONFIDENCE_CAP)
    return score


def validate(
    raw: Any,
    passage: str,
    dsld_id: int | str,
    *,
    name: str | None = None,
    config_hash: str | None = None,
    provenance: str = "local",
    latency_ms: int | None = None,
    error: str | None = None,
    status: str | None = None,
) -> dict:
    """Turn one model result into one row, grounded, capped and statused.

    `raw` is the model's text, or the dict it parsed to. `latency_ms` stays
    None for the agent driver, whose wall-clock is not comparable to inference
    latency. Pass `status="error"` with `error` set when the call itself failed.
    """
    raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)

    row = {
        "dsld_id": int(dsld_id),
        "name": name,
        "purpose": [],
        "risk": [],
        "purpose_confidence": 0,
        "risk_confidence": 0,
        "status": status or "ok",
        "error": error,
        "raw": raw_text,
        "latency_ms": latency_ms,
        "provenance": provenance,
        "config_hash": config_hash,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    if status in {"error", "poison"}:
        return row

    data: Any = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            row["status"] = "unparseable"
            return row
    if not isinstance(data, dict):
        row["status"] = "unparseable"
        return row

    purpose = _entries(data.get("purpose"), passage)
    risk = _entries(data.get("risk"), passage)

    row["purpose"] = purpose
    row["risk"] = risk
    row["purpose_confidence"] = _confidence(data.get("purpose_confidence"), purpose)
    row["risk_confidence"] = _confidence(data.get("risk_confidence"), risk)
    # Correctly finding nothing is a result, not a failure.
    row["status"] = "ok" if (purpose or risk) else "empty"
    return row


def grounding_rate(rows: list[dict]) -> float | None:
    """Share of entries whose evidence occurs verbatim in its passage.

    The corruption alarm: this architecture has a known CUDA bug that makes the
    model load normally and emit fluent garbage, and a fluent fabrication
    cannot reproduce exact substrings. None when there are no entries to judge.
    """
    entries = [e for r in rows for e in (*r.get("purpose", []), *r.get("risk", []))]
    if not entries:
        return None
    return sum(1 for e in entries if e["grounded"]) / len(entries)
