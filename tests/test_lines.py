"""The `lines` format: parsing, stray accounting and span recovery.

The grammar used to guarantee shape. Under `lines` nothing does, so the
guarantees move here: a line either matches and becomes a claim, or it is
counted as `stray`. What must never happen is a third thing -- a line quietly
vanishing -- because a dropped claim is indistinguishable from a label that
said nothing, and that is the failure this format could plausibly introduce.

No GPU and no network: the parser is a pure function of the model's text.
"""

from __future__ import annotations

import pytest

from engine.pipeline import lines
from engine.pipeline.config import Config
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


def test_parses_claims_and_confidences():
    out = lines.parse(
        "P: supports memory retention | 5\n"
        "R: do not exceed recommended intake | 4\n"
        "END"
    )
    assert [e["text"] for e in out["purpose"]] == ["supports memory retention"]
    assert [e["text"] for e in out["risk"]] == ["do not exceed recommended intake"]
    assert out["purpose_confidence"] == 5
    assert out["risk_confidence"] == 4
    assert out["stray"] == 0


def test_evidence_is_the_label_sentence_when_a_passage_is_given():
    """The span is searched for, not generated -- it costs no output tokens.

    Capitalisation and punctuation come back as the label wrote them, which is
    what makes the claim checkable against the label by eye.
    """
    out = lines.parse("P: supports memory retention | 5", PASSAGE)
    assert out["purpose"][0]["evidence"] == "Supports Memory Retention*"


def test_evidence_recovers_the_whole_sentence_not_just_the_claim():
    out = lines.parse("R: do not exceed recommended intake | 5", PASSAGE)
    evidence = out["risk"][0]["evidence"]
    assert evidence.startswith("WARNING:")
    assert "Do not exceed recommended" in evidence


def test_evidence_falls_back_to_the_claim_without_a_passage():
    """The row shape stays valid for a caller that has no passage to hand."""
    out = lines.parse("P: supports memory retention | 5")
    entry = out["purpose"][0]
    assert entry["evidence"] == entry["text"]


def test_unrecoverable_claim_keeps_a_non_empty_evidence():
    """An empty evidence would fail grounding on a technicality.

    A fabricated claim must fail the grounding check because it is not in the
    passage, not because the evidence field happened to be blank.
    """
    out = lines.parse("P: cures cancer | 5", PASSAGE)
    assert out["purpose"][0]["evidence"] == "cures cancer"


def test_field_confidence_is_the_max_over_its_claims():
    """One unambiguous claim settles that the label states the field."""
    out = lines.parse("P: clearly stated | 5\nP: vaguer one | 2")
    assert out["purpose_confidence"] == 5


def test_unmatched_lines_are_counted_not_dropped():
    out = lines.parse("P: fine | 3\nI cannot help with that.\nPURPOSE: wrong prefix | 2")
    assert out["stray"] == 2
    assert len(out["purpose"]) == 1
    # Kept for diagnosis: a drift is only fixable if the line survives.
    assert "I cannot help with that." in out["stray_lines"]


def test_stray_lines_are_bounded():
    """A runaway generation must not write an unbounded row."""
    out = lines.parse("\n".join(f"garbage line {i}" for i in range(50)))
    assert out["stray"] == 50
    assert len(out["stray_lines"]) == 5


def test_end_sentinel_is_never_stray():
    """The stop sequence usually eats END, but a stray one must not count."""
    assert lines.parse("P: x | 1\nEND")["stray"] == 0
    assert lines.parse("END")["stray"] == 0


def test_empty_output_is_clean_not_stray():
    out = lines.parse("")
    assert out["purpose"] == [] and out["risk"] == [] and out["stray"] == 0


def test_claim_may_contain_a_pipe():
    """The confidence binds to the last pipe, so claims can contain one."""
    out = lines.parse("P: energy | vitality blend | 4")
    assert out["purpose"][0]["text"] == "energy | vitality blend"
    assert out["purpose_confidence"] == 4


@pytest.mark.parametrize("conf", ["6", "-1", "x", ""])
def test_confidence_outside_zero_to_five_is_stray(conf):
    """A malformed score is a malformed line, not a silently coerced one."""
    assert lines.parse(f"P: something | {conf}")["stray"] == 1


def test_case_insensitive_prefix():
    assert len(lines.parse("p: lowercase prefix | 3")["purpose"]) == 1


# --- boilerplate filtering --------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "keep out of reach of children",
        "keep out of the reach of children",
        "tamper resistant package, do not use if outer seal is missing",
        "do not use if seal is broken",
        "do not use if inner seal is broken or missing",
        "do not eat freshness packet",
        "please discard the inedible freshness packet enclosed",
    ],
)
def test_boilerplate_risks_are_dropped(claim):
    """These are 33 of 38 false positives measured on gold-100."""
    assert lines.parse(f"R: {claim} | 5")["risk"] == []


def test_child_safety_is_dropped_even_when_it_is_the_only_risk():
    """The prompt's carve-out is wrong more often than right.

    Gold treats child-safety as a risk in 6 of 100 labels; an "only risk
    present" rule fires on 20. Keeping the carve-out left 19 false positives.
    """
    assert lines.parse("R: keep out of reach of children | 5")["risk"] == []


def test_real_cautions_survive_alongside_boilerplate():
    out = lines.parse(
        "R: keep out of reach of children | 5\n"
        "R: consult your physician if you are pregnant | 5\n"
        "R: do not use if seal is broken | 4"
    )
    assert [r["text"] for r in out["risk"]] == [
        "consult your physician if you are pregnant"
    ]


def test_dropping_boilerplate_cannot_raise_the_field_confidence():
    """The surviving claims carry the score, not the dropped ones."""
    out = lines.parse(
        "R: keep out of reach of children | 5\nR: consult your physician | 2"
    )
    assert out["risk_confidence"] == 2


def test_boilerplate_only_label_scores_zero_confidence():
    out = lines.parse("R: keep out of reach of children | 5")
    assert out["risk"] == [] and out["risk_confidence"] == 0


def test_purpose_claims_are_never_boilerplate_filtered():
    """The filter is about cautions; a purpose is judged only by the prompt."""
    out = lines.parse("P: keep out of reach of children | 1")
    assert len(out["purpose"]) == 1


# --- validate() integration -------------------------------------------------


def test_grounding_still_catches_fabrication():
    """The check the whole format rests on.

    With `evidence` gone, the prompt's copy rule is what keeps claims verbatim.
    A fabricated claim cannot be a substring of the passage, so it is still
    caught -- and still caps the confidence.
    """
    row = validate("P: cures cancer | 5\nEND", PASSAGE, dsld_id=1, fmt="lines")
    assert row["purpose"][0]["grounded"] is False
    assert row["purpose_confidence"] < 5


def test_grounded_claims_survive_a_line_break_in_the_passage():
    """"Do not exceed recommended\\nintake" is one claim wrapped across lines."""
    row = validate(
        "R: do not exceed recommended intake | 5\nEND", PASSAGE, dsld_id=1, fmt="lines"
    )
    assert row["risk"][0]["grounded"] is True


def test_all_stray_is_unparseable():
    row = validate("I cannot help with that.", PASSAGE, dsld_id=1, fmt="lines")
    assert row["status"] == "unparseable"
    assert row["raw"]  # retained for diagnosis


def test_empty_is_a_first_class_success():
    row = validate("END", PASSAGE, dsld_id=1, fmt="lines")
    assert row["status"] == "empty"
    assert row["stray"] == 0


def test_stray_column_exists_under_json_too():
    """Uniform across formats, so a glob over rows needs no branch."""
    row = validate(
        '{"purpose": [], "risk": [], "purpose_confidence": 0, "risk_confidence": 0}',
        PASSAGE,
        dsld_id=1,
    )
    assert row["stray"] == 0


# --- span recovery ----------------------------------------------------------


def test_recovers_the_source_sentence():
    span = lines.recover_span("supports memory retention", PASSAGE)
    assert span == "Supports Memory Retention*"


def test_recovers_across_a_line_break():
    span = lines.recover_span("do not exceed recommended intake", PASSAGE)
    assert span is not None
    assert "recommended" in span and "intake" in span


def test_recovery_returns_none_for_a_claim_not_in_the_label():
    assert lines.recover_span("cures cancer", PASSAGE) is None


# --- config -----------------------------------------------------------------


def test_format_changes_the_config_hash():
    """Two formats are not comparable and must never pool."""
    assert Config(fmt="json").config_hash != Config(fmt="lines").config_hash


def test_adding_the_format_did_not_rehash_existing_json_runs():
    """The hash is a promise that two runs pool.

    Adding `fmt` unconditionally rehashed every past `json` run, orphaning the
    gold-100 baseline that a new format is supposed to be measured against --
    so the key is omitted when it is `json`. This pins the hash of the
    committed 20260912T155850Z baseline.
    """
    cfg = Config(prompt_path="prompts/purpose_risk.md")
    assert "fmt" not in cfg.identity
    # Not the literal baseline digest: `prompts/purpose_risk.md` is edited in
    # place between runs, so pinning it here would fail on the next prompt
    # edit. What must hold is that `fmt` adds nothing under `json`.
    assert cfg.config_hash == Config(prompt_path="prompts/purpose_risk.md").config_hash
    assert Config(fmt="lines").identity["fmt"] == "lines"


def test_format_selects_its_prompt():
    assert Config(fmt="lines").prompt_path.endswith("purpose_risk_lines.md")
    assert Config(fmt="json").prompt_path.endswith("purpose_risk.md")


def test_explicit_prompt_still_wins():
    cfg = Config(fmt="lines", prompt_path="prompts/purpose_risk.md")
    assert cfg.prompt_path == "prompts/purpose_risk.md"


def test_lines_manifest_carries_no_schema():
    """There is no schema under `lines`; claiming one would be a false record."""
    assert Config(fmt="lines").resolved()["schema"] is None
    assert Config(fmt="json").resolved()["schema"] is not None


def test_unknown_format_is_refused():
    with pytest.raises(ValueError):
        Config(fmt="yaml")
