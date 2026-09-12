"""Validation is the highest deterministic seam in the pipeline.

Grounding, the confidence cap and status assignment all land here, and none of
them needs a GPU or a network. Fixtures are small hand-written dicts rather
than slices of the real gold set, so regenerating the gold set cannot break
these tests.
"""

from __future__ import annotations

import pytest

from engine.pipeline.validate import validate

PASSAGE = (
    "PRODUCT: Huperzine A\n"
    "BRAND: GNC Herbal Plus Standardized\n"
    "\n"
    "LABEL STATEMENTS:\n"
    "[General Statements: All Other Content] Supports Memory Retention*\n"
    "[Precautions re: All Other] WARNING: Do not exceed recommended\n"
    "intake. Discontinue use two weeks prior to surgery."
)


def _raw(**over):
    raw = {
        "purpose": [
            {"text": "supports memory retention", "evidence": "Supports Memory Retention*"}
        ],
        "risk": [],
        "purpose_confidence": 5,
        "risk_confidence": 0,
    }
    raw.update(over)
    return raw


def test_evidence_present_in_passage_is_grounded():
    row = validate(_raw(), PASSAGE, dsld_id=709)

    assert row["status"] == "ok"
    assert row["purpose"][0]["grounded"] is True
    assert row["purpose_confidence"] == 5


def test_evidence_differing_only_in_whitespace_is_grounded():
    """A model quoting faithfully across a line break is not a fabrication."""
    raw = _raw(
        risk=[
            {
                "text": "do not exceed recommended intake",
                # The passage breaks this span across a newline.
                "evidence": "WARNING: Do not exceed recommended intake.",
            }
        ],
        risk_confidence=4,
    )

    row = validate(raw, PASSAGE, dsld_id=709)

    assert row["risk"][0]["grounded"] is True
    assert row["risk_confidence"] == 4


def test_evidence_absent_from_passage_is_ungrounded():
    raw = _raw(
        purpose=[
            {"text": "cures insomnia", "evidence": "Clinically proven to cure insomnia"}
        ]
    )

    row = validate(raw, PASSAGE, dsld_id=709)

    assert row["purpose"][0]["grounded"] is False


def test_ungrounded_evidence_caps_a_confident_score():
    """A confident hallucination must not be able to score itself highly."""
    raw = _raw(
        purpose=[
            {"text": "cures insomnia", "evidence": "Clinically proven to cure insomnia"}
        ],
        purpose_confidence=5,
    )

    row = validate(raw, PASSAGE, dsld_id=709)

    assert row["purpose_confidence"] == 1


def test_one_ungrounded_entry_caps_the_whole_field():
    raw = _raw(
        purpose=[
            {"text": "supports memory retention", "evidence": "Supports Memory Retention*"},
            {"text": "cures insomnia", "evidence": "Clinically proven to cure insomnia"},
        ],
        purpose_confidence=5,
    )

    row = validate(raw, PASSAGE, dsld_id=709)

    assert row["purpose_confidence"] == 1


def test_empty_extraction_scores_zero_confidence():
    """"Found nothing" is never a confident answer, whatever was claimed."""
    row = validate(_raw(purpose=[], purpose_confidence=4), PASSAGE, dsld_id=709)

    assert row["purpose"] == []
    assert row["purpose_confidence"] == 0


def test_both_fields_empty_is_status_empty_not_a_failure():
    """28% of labels state no purpose and 29% no risk; that is a success."""
    row = validate(
        _raw(purpose=[], risk=[], purpose_confidence=0, risk_confidence=0),
        PASSAGE,
        dsld_id=709,
    )

    assert row["status"] == "empty"


def test_one_field_populated_is_ok_not_empty():
    row = validate(_raw(risk=[]), PASSAGE, dsld_id=709)

    assert row["status"] == "ok"


def test_unparseable_raw_text_is_status_unparseable_and_keeps_the_text():
    """An unparseable result must be diagnosable after the fact."""
    row = validate("not json at all {{", PASSAGE, dsld_id=709)

    assert row["status"] == "unparseable"
    assert row["raw"] == "not json at all {{"
    assert row["purpose"] == []
    assert row["risk"] == []


def test_wrong_typed_confidence_does_not_crash_validation():
    row = validate(_raw(purpose_confidence="high"), PASSAGE, dsld_id=709)

    assert row["status"] in {"ok", "unparseable"}
    assert isinstance(row["purpose_confidence"], int)


def test_a_json_string_is_parsed_like_a_dict():
    import json

    row = validate(json.dumps(_raw()), PASSAGE, dsld_id=709)

    assert row["status"] == "ok"
    assert row["purpose"][0]["grounded"] is True


def test_row_carries_identity_and_provenance_columns():
    """A row must be interpretable without joining it back to a manifest."""
    row = validate(
        _raw(), PASSAGE, dsld_id=709, config_hash="abc123", provenance="local"
    )

    assert row["dsld_id"] == 709
    assert row["config_hash"] == "abc123"
    assert row["provenance"] == "local"


def test_latency_is_nullable_for_the_agent_driver():
    row = validate(_raw(), PASSAGE, dsld_id=709, provenance="agent")

    assert row["latency_ms"] is None


@pytest.mark.parametrize("status", ["ok", "empty", "unparseable"])
def test_every_row_keeps_raw_text_whatever_the_status(status):
    raws = {
        "ok": _raw(),
        "empty": _raw(purpose=[], risk=[]),
        "unparseable": "{oops",
    }

    row = validate(raws[status], PASSAGE, dsld_id=709)

    assert row["status"] == status
    assert row["raw"]
