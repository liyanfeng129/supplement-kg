"""A broken scorer yields plausible numbers and misdirects a week of tuning.

`smoke` catches a broken runtime; nothing catches a broken scorer except
tests, which is why these exist despite the pipeline's other IO stages being
covered only transitively. Fixtures are hand-written, not sliced from
`datasets/pilot/gold-100.jsonl`, so regenerating the gold set cannot break them.
"""

from __future__ import annotations

from engine.pipeline.score import score


def _row(dsld_id, purpose=(), risk=(), status="ok"):
    """A predicted row, in the shape `validate()` writes."""
    return {
        "dsld_id": dsld_id,
        "status": status,
        "purpose": [{"text": t, "evidence": e, "grounded": True} for t, e in purpose],
        "risk": [{"text": t, "evidence": e, "grounded": True} for t, e in risk],
        "purpose_confidence": 5 if purpose else 0,
        "risk_confidence": 5 if risk else 0,
    }


def _gold(dsld_id, purpose=(), risk=()):
    return {
        "dsld_id": dsld_id,
        "purpose": [{"text": t, "evidence": e, "grounded": True} for t, e in purpose],
        "risk": [{"text": t, "evidence": e, "grounded": True} for t, e in risk],
    }


SUPPORTS = ("supports memory retention", "Supports Memory Retention*")
WARNING = ("do not exceed recommended intake", "WARNING: Do not exceed recommended intake.")


def test_exact_match_scores_a_hit_on_both_axes():
    gold = [_gold(1, purpose=[SUPPORTS], risk=[WARNING])]
    rows = [_row(1, purpose=[SUPPORTS], risk=[WARNING])]

    m = score(rows, gold)

    assert m["presence"]["accuracy"] == 1.0
    assert m["evidence"]["precision"] == 1.0
    assert m["evidence"]["recall"] == 1.0
    assert m["hallucination_rate"] == 0.0


def test_a_paraphrase_citing_the_same_evidence_still_matches():
    """Detail matching is on evidence spans, not on `text` token overlap.

    A rephrased claim and a missed claim demand opposite prompt changes, so
    the metric must not conflate them.
    """
    gold = [_gold(1, purpose=[SUPPORTS])]
    rows = [_row(1, purpose=[("helps with memory", "Supports Memory Retention*")])]

    m = score(rows, gold)

    assert m["presence"]["accuracy"] == 1.0
    assert m["evidence"]["recall"] == 1.0


def test_a_missed_claim_costs_recall_not_precision():
    gold = [_gold(1, purpose=[SUPPORTS, ("supports focus", "Supports Focus*")])]
    rows = [_row(1, purpose=[SUPPORTS])]

    m = score(rows, gold)

    assert m["evidence"]["recall"] == 0.5
    assert m["evidence"]["precision"] == 1.0


def test_a_hallucinated_claim_costs_precision_not_recall():
    gold = [_gold(1, purpose=[SUPPORTS])]
    rows = [_row(1, purpose=[SUPPORTS, ("cures insomnia", "Clinically proven to cure insomnia")])]

    m = score(rows, gold)

    assert m["evidence"]["precision"] == 0.5
    assert m["evidence"]["recall"] == 1.0


def test_correctly_empty_is_a_hit_not_a_miss():
    """The headline is presence/absence: agreeing on "nothing here" is correct."""
    gold = [_gold(1, purpose=[], risk=[])]
    rows = [_row(1, purpose=[], risk=[], status="empty")]

    m = score(rows, gold)

    assert m["presence"]["accuracy"] == 1.0
    assert m["presence"]["true_negatives"] == 2


def test_asserting_a_purpose_where_gold_says_none_is_a_hallucination():
    gold = [_gold(1, purpose=[], risk=[])]
    rows = [_row(1, purpose=[("cures insomnia", "cures insomnia")])]

    m = score(rows, gold)

    assert m["hallucination_rate"] > 0
    assert m["presence"]["accuracy"] == 0.5


def test_missing_a_field_gold_states_is_not_counted_as_hallucination():
    gold = [_gold(1, purpose=[SUPPORTS], risk=[WARNING])]
    rows = [_row(1, purpose=[], risk=[], status="empty")]

    m = score(rows, gold)

    assert m["hallucination_rate"] == 0.0
    assert m["presence"]["accuracy"] == 0.0


def test_headline_counts_two_cells_per_label():
    gold = [_gold(1, purpose=[SUPPORTS], risk=[]), _gold(2, purpose=[], risk=[WARNING])]
    rows = [_row(1, purpose=[SUPPORTS], risk=[]), _row(2, purpose=[], risk=[WARNING])]

    m = score(rows, gold)

    assert m["presence"]["n_cells"] == 4
    assert m["n_labels"] == 2


def test_claims_sharing_one_evidence_block_hit_the_documented_ceiling():
    """29% of gold risk entries share an evidence block with a sibling.

    Span matching cannot separate three claims citing one WARNING block from
    one claim citing it. The scorer must report that as a documented ceiling,
    not silently penalise the model for it.
    """
    block = "WARNING: Do not exceed intake. Discontinue two weeks before surgery."
    gold = [
        _gold(
            1,
            risk=[("do not exceed intake", block), ("discontinue before surgery", block)],
        )
    ]
    rows = [_row(1, risk=[("do not exceed intake", block), ("discontinue before surgery", block)])]

    m = score(rows, gold)

    assert m["evidence"]["recall"] == 1.0
    assert m["evidence"]["precision"] == 1.0
    assert m["evidence"]["shared_evidence_claims"] == 2


def test_rows_absent_from_the_run_are_reported_not_silently_skipped():
    gold = [_gold(1, purpose=[SUPPORTS]), _gold(2, purpose=[SUPPORTS])]
    rows = [_row(1, purpose=[SUPPORTS])]

    m = score(rows, gold)

    assert m["n_labels"] == 1
    assert m["n_gold_missing_from_run"] == 1


def test_non_ok_rows_do_not_count_as_confident_absence():
    """An errored row is machinery breaking, not the model finding nothing."""
    gold = [_gold(1, purpose=[SUPPORTS])]
    rows = [_row(1, purpose=[], status="error")]

    m = score(rows, gold)

    assert m["n_labels"] == 0
    assert m["n_excluded_not_ok"] == 1


def test_the_reference_caveat_travels_with_the_metrics():
    """The gold set is one model's reading, not human adjudication."""
    m = score([_row(1, purpose=[SUPPORTS])], [_gold(1, purpose=[SUPPORTS])])

    assert "caveat" in m
    assert "adjudication" in m["caveat"].lower()


def test_a_trivially_short_evidence_span_does_not_match_everything():
    """Unbounded containment would let `evidence: "the"` hit every gold span."""
    gold = [_gold(1, purpose=[SUPPORTS])]
    rows = [_row(1, purpose=[("supports memory retention", "up")])]

    m = score(rows, gold)

    assert m["evidence"]["precision"] == 0.0
    assert m["evidence"]["recall"] == 0.0


def test_a_tighter_sub_span_of_the_gold_block_still_matches():
    """A more precise citation of the same text is a hit, not a miss."""
    block = "WARNING: Do not exceed recommended intake. Consult your physician."
    gold = [_gold(1, risk=[("do not exceed intake", block)])]
    rows = [_row(1, risk=[("do not exceed intake", "Do not exceed recommended intake.")])]

    m = score(rows, gold)

    assert m["evidence"]["precision"] == 1.0


def test_hallucination_is_also_reported_conditional_on_gold_being_empty():
    """The share of all cells is diluted; the conditional rate is the signal.

    One label states a purpose and no risk; the model invents a risk. That is
    every empty-gold cell hallucinated, but only a quarter of all cells.
    """
    gold = [_gold(1, purpose=[SUPPORTS], risk=[]), _gold(2, purpose=[SUPPORTS], risk=[])]
    rows = [
        _row(1, purpose=[SUPPORTS], risk=[("invented", "Clinically proven to cure all")]),
        _row(2, purpose=[SUPPORTS], risk=[("invented", "Clinically proven to cure all")]),
    ]

    m = score(rows, gold)

    assert m["n_gold_empty_cells"] == 2
    assert m["hallucination_rate_given_empty"] == 1.0
    assert m["hallucination_rate"] == 0.5
