"""The human sheet (spec 0011.04): blind cases, checked saves, appended verdicts.

What is pinned: `sheet_cases` hands out only the pilot's `in_human` labels and those in `also`, in
`sha256(case_id)` order, as the passage and its claims with no stratum or
type; `save` stores nothing and names the problem when a claim is judged
other than once, a field is not one of its values, `actual_kind` is set on an
invented claim or missing on another, the active time is not a whole number of
seconds, or a missed quote is not in the passage
or overlaps a claim. A label saved twice reads back as its last save; a torn
last line is skipped on read and cut before the next append.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from engine.repositories.dsld.labels import loader as label_loader
from engine.repositories.dsld.labels import paths as label_paths
from engine.repositories.dsld.labels import store as label_store
from engine.repositories.dsld.runs import loader as runs
from engine.repositories.dsld.runs import paths as run_paths
from engine.repositories.dsld.runs import store as run_store
from engine.repositories.dsld.runs.schema import STRATA, ClaimVerdict, HumanVerdict, MissedClaim
from engine.services.extraction_eval.human_sheet import save, sheet_cases
from engine.services.extraction_eval.profile import build_profile
from engine.services.extraction_eval.sample import build_pilot
from engine.services.extraction_eval.schema import Quota

PASSAGE = "do not take if pregnant\nsupports healthy joints and eases stiffness\nkeep out of reach of children"

ROWS = [
    {"dsld_id": 1, "status": "ok", "purpose": [{"text": "supports healthy joints"}], "risk": [{"text": "do not take if pregnant"}]},
    {"dsld_id": 2, "status": "ok", "purpose": [{"text": "eases stiffness"}], "risk": [{"text": "do not take if pregnant"}]},
    {"dsld_id": 3, "status": "ok", "purpose": [{"text": "supports healthy joints"}], "risk": [{"text": "keep out of reach"}]},
    *({"dsld_id": i, "status": "ok", "purpose": [], "risk": []} for i in (6, 7, 8)),
]

# Labels 1 to 3 are all `both, verbatim`: 3 in the pilot, 2 of them for the human; one `neither`, not.
QUOTAS = {s: Quota(0, 0, 0, 0) for s in STRATA}
QUOTAS[("both", "verbatim")] = Quota(3, 2, 0, 0)
QUOTAS[("neither", None)] = Quota(1, 0, 0, 0)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A profiled run over label set `tiny`, with its pilot drawn."""
    monkeypatch.setattr(label_paths, "LABELS_ROOT", tmp_path / "labels")
    monkeypatch.setattr(run_paths, "RUNS_ROOT", tmp_path / "runs")
    label_loader.reset()
    ids = [r["dsld_id"] for r in ROWS]
    sha = label_store.write_label_set(
        "tiny", [{"dsld_id": i, "name": f"label {i}", "passage": PASSAGE} for i in ids], {}
    )
    d = run_store.new_run("tiny", ids, {"label_set": "tiny", "passages_sha": sha, "config_hash": "abc"})
    run_paths.rows_path(d).write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    build_profile(d)
    build_pilot(d, QUOTAS, 7)
    yield d
    label_loader.reset()


def _verdict(case, **override):
    """A verdict on `case` that passes: every claim faithful, of its own kind."""
    fields = {
        "case_id": case.case_id,
        "dsld_id": case.dsld_id,
        "claims": [ClaimVerdict(c.claim_id, "yes", c.kind, False, "") for c in case.claims],
        "missed": [],
        "opened_at": "2026-10-27T10:00:00+00:00",
        "saved_at": "2026-10-27T10:01:00+00:00",
        "active_seconds": 42,
    }
    fields.update(override)
    return HumanVerdict(**fields)


def test_only_the_pilot_s_human_part_in_hashed_order(run):
    cases = sheet_cases(run)
    human = [c.label.case_id for c in runs.read_pilot(run) if c.label.in_human]

    assert sorted(c.case_id for c in cases) == sorted(human)
    assert [c.case_id for c in cases] == sorted(human, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    case = cases[0]
    assert case.passage == PASSAGE
    assert [(c.kind, c.claim_id.rsplit(":", 2)[1]) for c in case.claims] == [("purpose", "purpose"), ("risk", "risk")]


def test_labels_in_also_join_the_human_part_in_one_hashed_order(run):
    pilot = runs.read_pilot(run)
    other = next(c.label.case_id for c in pilot if not c.label.in_human)
    cases = sheet_cases(run, also=[other])
    expected = [c.label.case_id for c in pilot if c.label.in_human] + [other]

    assert [c.case_id for c in cases] == sorted(expected, key=lambda i: hashlib.sha256(i.encode()).hexdigest())


def test_a_label_in_also_outside_the_pilot_raises(run):
    with pytest.raises(ValueError, match="not pilot labels"):
        sheet_cases(run, also=["label:99"])


def test_a_changed_label_set_raises(run):
    manifest_p = run_paths.manifest_path(run)
    manifest = json.loads(manifest_p.read_text())
    manifest_p.write_text(json.dumps({**manifest, "passages_sha": "0" * 16}))
    with pytest.raises(ValueError, match="passages_sha"):
        sheet_cases(run)


def test_a_passing_verdict_is_stored_and_read_back(run):
    case = sheet_cases(run)[0]
    verdict = _verdict(case, missed=[MissedClaim("eases stiffness", "purpose", True, "maybe")])

    assert save(run, case, verdict) == []
    assert runs.read_human_verdicts(run) == {case.case_id: verdict}


def test_the_last_save_of_a_label_wins(run):
    case = sheet_cases(run)[0]
    first, second = _verdict(case), _verdict(case, saved_at="2026-10-27T11:00:00+00:00")
    save(run, case, first)
    save(run, case, second)

    assert runs.read_human_verdicts(run)[case.case_id] == second
    assert len(run_paths.eva_path(run, run_paths.HUMAN_PILOT_NAME).read_text().splitlines()) == 2


@pytest.mark.parametrize(
    "change, problem",
    [
        (lambda c, v: {"claims": v.claims[:1]}, "judged 0 times"),
        (lambda c, v: {"claims": v.claims + v.claims[:1]}, "judged 2 times"),
        (lambda c, v: {"claims": v.claims + [ClaimVerdict("claim:99:risk:0", "yes", "risk", False, "")]}, "not a claim of this label"),
        (lambda c, v: {"claims": [ClaimVerdict(v.claims[0].claim_id, "maybe", "purpose", False, "")] + v.claims[1:]}, "faithful 'maybe'"),
        (lambda c, v: {"claims": [ClaimVerdict(v.claims[0].claim_id, "invented", "purpose", False, "")] + v.claims[1:]}, "invented, so actual_kind must be empty"),
        (lambda c, v: {"claims": [ClaimVerdict(v.claims[0].claim_id, "yes", None, False, "")] + v.claims[1:]}, "actual_kind None"),
        (lambda c, v: {"missed": [MissedClaim("cures everything", "purpose", False, "")]}, "not a piece of the passage"),
        (lambda c, v: {"missed": [MissedClaim("eases stiffness", "side_effect", False, "")]}, "kind 'side_effect'"),
        (lambda c, v: {"missed": [MissedClaim("healthy joints", "purpose", False, "")]}, "overlaps"),
        (lambda c, v: {"missed": [MissedClaim("supports healthy joints and eases stiffness", "purpose", False, "")]}, "overlaps"),
        (lambda c, v: {"case_id": "label:6", "dsld_id": 6}, "not label:"),
        (lambda c, v: {"active_seconds": -1}, "active_seconds -1"),
        (lambda c, v: {"active_seconds": 4.5}, "active_seconds 4.5"),
    ],
)
def test_a_failing_verdict_is_not_stored(run, change, problem):
    case = next(c for c in sheet_cases(run) if c.claims[0].text == "supports healthy joints")
    verdict = _verdict(case)
    problems = save(run, case, _verdict(case, **change(case, verdict)))

    assert any(problem in p for p in problems), problems
    assert runs.read_human_verdicts(run) == {}


def test_an_invented_claim_without_a_kind_passes(run):
    case = sheet_cases(run)[0]
    claims = [ClaimVerdict(case.claims[0].claim_id, "invented", None, False, "")] + _verdict(case).claims[1:]
    assert save(run, case, _verdict(case, claims=claims)) == []


def test_a_line_holding_a_purpose_and_a_risk_passes_as_both(run):
    case = sheet_cases(run)[0]
    claims = [ClaimVerdict(case.claims[0].claim_id, "yes", "both", False, "")] + _verdict(case).claims[1:]
    assert save(run, case, _verdict(case, claims=claims)) == []


def test_a_torn_last_line_is_skipped_then_cut_before_the_next_save(run):
    case = sheet_cases(run)[0]
    save(run, case, _verdict(case))
    path = run_paths.eva_path(run, run_paths.HUMAN_PILOT_NAME)
    with path.open("a") as fh:
        fh.write('{"case_id": "label:')

    assert list(runs.read_human_verdicts(run)) == [case.case_id]
    later = _verdict(case, saved_at="2026-10-27T12:00:00+00:00")
    save(run, case, later)
    assert runs.read_human_verdicts(run)[case.case_id] == later
    assert len(path.read_text().splitlines()) == 2
