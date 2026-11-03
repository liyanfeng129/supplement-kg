"""The human sheet's workflow: the labels it shows, and the checks a verdict passes before it is stored (spec 0011.04).

**Blind.** `sheet_cases` hands out the pilot's human part, and the pilot labels
the judges route to the human, as the passage and its claims only -- no stratum, group, claim type or rank -- ordered by
`sha256(case_id)`, because the pilot's stored order is stratum order and
would give the strata away.

**Checked, then stored.** `save` checks a verdict against the case it was
given on and stores it only when nothing is wrong; otherwise it returns what
is wrong and stores nothing. The checks are 0011.03's: every claim judged
exactly once, every field one of its values, `actual_kind` empty exactly when
`invented`, the active time a whole number of seconds, and every missed quote a
piece of the passage that overlaps no claim -- a second claim inside a claim is
part of that claim, not missed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, get_args

from engine.repositories.dsld.labels import loader as label_sets
from engine.repositories.dsld.runs import loader as runs_loader
from engine.repositories.dsld.runs import store as runs_store
from engine.repositories.dsld.runs.schema import ActualKind, Faithful, HumanVerdict, Kind
from engine.services.extraction_eval.schema import SheetCase, SheetClaim


def sheet_cases(run_dir: str | Path, also: Iterable[str] = ()) -> list[SheetCase]:
    """The pilot's `in_human` labels, and the pilot labels in `also`, as the sheet shows them, in `sha256(case_id)` order.

    `also` is case ids of pilot labels outside the human part that the human
    judges too: those the judges' verdicts route to them. They are mixed into
    the one hashed order, so nothing shows which label came in which way.

    Raises if the run has no pilot or a stale one (`read_pilot`), if a case id
    in `also` is not in the pilot, and if the run's label set is not the one
    the run was made on: its `passages_sha` differs from the run manifest's.
    """
    pilot = runs_loader.read_pilot(run_dir)
    also = set(also)
    unknown = also - {c.label.case_id for c in pilot}
    if unknown:
        raise ValueError(f"not pilot labels of {run_dir}: {sorted(unknown)}")

    # The run's label set, the one its manifest names and pins.
    manifest = runs_loader.read_manifest(run_dir)
    passages_sha, labels = label_sets.load_flattened(manifest["label_set"])
    if passages_sha != manifest["passages_sha"]:
        raise ValueError(
            f"label set {manifest['label_set']!r} has passages_sha {passages_sha} but run {run_dir} "
            f"was made on {manifest['passages_sha']}: the passages are not the ones it saw"
        )

    # The human part of the pilot and the labels in `also`, joined with their passages, blind.
    cases = [
        SheetCase(
            case_id=c.label.case_id,
            dsld_id=c.label.dsld_id,
            passage=labels[c.label.dsld_id].passage,
            claims=[
                *(SheetClaim(x.claim_id, "purpose", x.text) for x in c.purpose),
                *(SheetClaim(x.claim_id, "risk", x.text) for x in c.risk),
            ],
        )
        for c in pilot
        if c.label.in_human or c.label.case_id in also
    ]
    return sorted(cases, key=lambda c: hashlib.sha256(c.case_id.encode()).hexdigest())


def save(run_dir: str | Path, case: SheetCase, verdict: HumanVerdict) -> list[str]:
    """Check `verdict` against `case`; store it if nothing is wrong. Returns what is wrong, `[]` once stored.

    `case` is the one `sheet_cases` handed out, passed back rather than read
    again: a read joins the whole run and takes seconds.
    """
    problems = []

    # The verdict is for this case.
    if verdict.case_id != case.case_id or verdict.dsld_id != case.dsld_id:
        problems.append(
            f"verdict is for {verdict.case_id} ({verdict.dsld_id}), not {case.case_id} ({case.dsld_id})"
        )

    # Every claim judged exactly once, and no other.
    expected = [c.claim_id for c in case.claims]
    given = [v.claim_id for v in verdict.claims]
    for claim_id in expected:
        n = given.count(claim_id)
        if n != 1:
            problems.append(f"{claim_id}: judged {n} times, not once")
    for claim_id in sorted(set(given) - set(expected)):
        problems.append(f"{claim_id}: not a claim of this label")

    # Every field one of its values; actual_kind empty exactly when invented.
    for v in verdict.claims:
        if v.faithful not in get_args(Faithful):
            problems.append(f"{v.claim_id}: faithful {v.faithful!r} is not one of {get_args(Faithful)}")
        if v.faithful == "invented":
            if v.actual_kind is not None:
                problems.append(f"{v.claim_id}: invented, so actual_kind must be empty, not {v.actual_kind!r}")
        elif v.actual_kind not in get_args(ActualKind):
            problems.append(f"{v.claim_id}: actual_kind {v.actual_kind!r} is not one of {get_args(ActualKind)}")
        if not isinstance(v.unsure, bool):
            problems.append(f"{v.claim_id}: unsure must be true or false")
    if not isinstance(verdict.active_seconds, int) or isinstance(verdict.active_seconds, bool) or verdict.active_seconds < 0:
        problems.append(f"active_seconds {verdict.active_seconds!r} is not a whole number of seconds")

    # Every missed quote a piece of the passage, of a kind, overlapping no claim.
    for m in verdict.missed:
        if not m.quote.strip():
            problems.append("a missed claim has an empty quote")
            continue
        if m.quote not in case.passage:
            problems.append(f"missed {m.quote!r}: not a piece of the passage")
        if m.kind not in get_args(Kind):
            problems.append(f"missed {m.quote!r}: kind {m.kind!r} is not one of {get_args(Kind)}")
        for c in case.claims:
            if m.quote in c.text or c.text in m.quote:
                problems.append(
                    f"missed {m.quote!r} overlaps {c.claim_id} ({c.text!r}): a second claim inside a "
                    f"claim is part of that claim, and a claim already extracted is not missed"
                )
        if not isinstance(m.unsure, bool):
            problems.append(f"missed {m.quote!r}: unsure must be true or false")

    if problems:
        return problems
    runs_store.save_human_verdict(run_dir, verdict)
    return []
