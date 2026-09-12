"""Rows plus gold to metrics.

**Headline: per-field presence/absence over 200 cells** (100 labels x 2
fields), with **hallucination rate** -- a field asserted where gold says none --
called out separately, because grounding over recall is the stated priority.
At 100 labels the headline is robust in a way a claim-level number is not.

**Detail: evidence-span precision and recall per claim**, matching on overlap
of verbatim spans rather than token overlap on `text`. Token overlap cannot
distinguish a missed claim from a rephrased one, and those demand opposite
prompt changes.

**The detail metric's ceiling.** 55 of 190 gold risk entries (29%) share an
evidence block with a sibling: three cautions packed into one WARNING block
cite that same block three times. Span matching therefore cannot separate
three claims citing one block from one claim citing it, and a model that
splits the block differently from gold scores the same either way. That 29% is
a **metric limitation, never model error**. `evidence.shared_evidence_claims`
reports how many gold claims sit under that ceiling in the set being scored, so
the number is read rather than inferred.

**The gold set is a reference, not ground truth.** It is one frontier model's
reading of the label -- auditable because evidence spans are kept, not a human
adjudication. Precision against it means *agreement with that reading*. That
caveat is returned in `caveat` so it travels with the numbers wherever they are
reported. 100 labels gives +-10pp at best: this is a screen for breakage and
large wins. When two prompts land within a few points the answer is more gold
labels -- drawn randomly from the remaining 900, since these 100 are the
lowest-ID decile and so age-biased -- not a finer metric on the same 100.
"""

from __future__ import annotations

from engine.services.extraction.label_purpose_risk import _normalise

FIELDS = ("purpose", "risk")

CAVEAT = (
    "The gold set is one frontier model's reading of the label, auditable via "
    "its evidence spans, not a human adjudication. Scores measure agreement "
    "with that reading. At 100 labels (+-10pp) this is a screen for breakage "
    "and large wins, not a ranking."
)


def _spans(entries: list[dict]) -> list[str]:
    """Normalised evidence spans, one per entry, blanks dropped."""
    return [_normalise(e.get("evidence") or "") for e in entries or [] if (e.get("evidence") or "").strip()]


# Shortest evidence span that may be matched by containment. Without a floor,
# a one-word `evidence` ("the") is contained in nearly every gold span and
# would score as a hit against all of them, inflating precision for exactly the
# degenerate output the metric should punish.
MIN_SPAN_CHARS = 12


def _matches(pred: str, gold_spans: list[str]) -> bool:
    """Whether a predicted span overlaps any gold span.

    Containment either way counts: a model citing a tighter sub-span of the
    block gold cited is pointing at the same text, which is what the span is
    for. Exact equality alone would fail a correct, more precise citation.

    Spans below `MIN_SPAN_CHARS` must match exactly, since containment of a very
    short string says nothing about whether the two cite the same text.
    """
    for g in gold_spans:
        if not pred or not g:
            continue
        if pred == g:
            return True
        if min(len(pred), len(g)) < MIN_SPAN_CHARS:
            continue
        if pred in g or g in pred:
            return True
    return False


def _shared_evidence_claims(gold_rows: list[dict]) -> int:
    """Gold claims whose evidence block is cited by a sibling claim too.

    These sit under the documented ceiling: span matching cannot tell them
    apart. Reported, not penalised.
    """
    n = 0
    for g in gold_rows:
        for fld in FIELDS:
            spans = _spans(g.get(fld, []))
            n += sum(1 for s in spans if spans.count(s) > 1)
    return n


def score(rows: list[dict], gold: list[dict]) -> dict:
    """Score a run's rows against the gold set.

    Only `ok` and `empty` rows are scored: an `error` or `unparseable` row is
    machinery breaking, not the model finding nothing, and counting it as a
    confident absence would flatter a broken run. Those are reported
    separately as `n_excluded_not_ok`.
    """
    by_id = {int(r["dsld_id"]): r for r in rows}
    scored: list[tuple[dict, dict]] = []
    excluded = 0
    missing = 0

    for g in gold:
        row = by_id.get(int(g["dsld_id"]))
        if row is None:
            missing += 1
            continue
        if row.get("status") not in {"ok", "empty"}:
            excluded += 1
            continue
        scored.append((row, g))

    tp = tn = fp = fn = 0
    ev_hits = ev_pred = ev_gold = 0

    for row, g in scored:
        for fld in FIELDS:
            pred_entries = row.get(fld) or []
            gold_entries = g.get(fld) or []
            has_pred, has_gold = bool(pred_entries), bool(gold_entries)

            if has_pred and has_gold:
                tp += 1
            elif not has_pred and not has_gold:
                tn += 1
            elif has_pred and not has_gold:
                fp += 1  # asserted a field the label does not state
            else:
                fn += 1

            pred_spans, gold_spans = _spans(pred_entries), _spans(gold_entries)
            ev_pred += len(pred_spans)
            ev_gold += len(gold_spans)
            ev_hits += sum(1 for p in pred_spans if _matches(p, gold_spans))

    n_cells = tp + tn + fp + fn
    # Recall is counted from the gold side: a gold span is recalled when some
    # prediction overlaps it, so two predictions citing one block do not score
    # it twice.
    ev_recalled = 0
    for row, g in scored:
        for fld in FIELDS:
            pred_spans = _spans(row.get(fld) or [])
            for gs in _spans(g.get(fld) or []):
                if _matches(gs, pred_spans):
                    ev_recalled += 1

    return {
        "n_labels": len(scored),
        "n_gold": len(gold),
        "n_gold_missing_from_run": missing,
        "n_excluded_not_ok": excluded,
        "presence": {
            "n_cells": n_cells,
            "accuracy": (tp + tn) / n_cells if n_cells else 0.0,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        },
        # A field asserted where gold says none. Reported two ways because they
        # answer different questions and are easy to confuse:
        #   `hallucination_rate`  - share of ALL cells that are false
        #       assertions. Comparable across runs, but diluted by the ~70% of
        #       cells where gold does state the field.
        #   `hallucination_rate_given_empty` - share of the cells where gold
        #       says nothing that the model filled in anyway. This is the
        #       conditional rate, and the one that moves when the model starts
        #       inventing claims.
        "hallucination_rate": fp / n_cells if n_cells else 0.0,
        "hallucination_rate_given_empty": fp / (fp + tn) if (fp + tn) else 0.0,
        "n_gold_empty_cells": fp + tn,
        "evidence": {
            "precision": ev_hits / ev_pred if ev_pred else 0.0,
            "recall": ev_recalled / ev_gold if ev_gold else 0.0,
            "n_predicted_claims": ev_pred,
            "n_gold_claims": ev_gold,
            # Gold claims sharing an evidence block with a sibling: the
            # documented ceiling, reported so it is read as a metric limit.
            "shared_evidence_claims": _shared_evidence_claims([g for _, g in scored]),
        },
        "caveat": CAVEAT,
    }
