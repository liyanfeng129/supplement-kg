"""The sample: a seeded, stratified draw of labels for the pilot, the large set and the human set (spec 0011.02).

What is pinned: a label's rarest type over both kinds, first match wins
(spread or invented, then stitched, then verbatim; none without claims); the
10 strata partition the profile's labels; within a stratum one seeded order,
`sha256(seed:dsld_id)`, of which every set is a prefix -- pilot human, pilot,
human, large -- independent of the run's row order; a quota above what is
left takes all of it; `draw` refuses a pilot that is not `draw_pilot`'s.
Quotas must name exactly the 10 strata and nest. The store refuses a second
save; a read joins each label with its claims' text and type, and raises when
the run's rows or profile changed under it.
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
from engine.repositories.dsld.runs.schema import STRATA, ProfiledLabel
from engine.services.extraction_eval.profile import build_profile
from engine.services.extraction_eval.sample import build_evaluation_set, build_pilot, draw, draw_pilot
from engine.services.extraction_eval.schema import Profile, Quota

SEED = 7


def _profile(*labels):
    return Profile(list(labels), {}, {}, [])


def _label(dsld_id, purpose=(), risk=()):
    group = (
        "both" if purpose and risk else "purpose_only" if purpose else "risk_only" if risk else "neither"
    )
    return ProfiledLabel(dsld_id, group, list(purpose), list(risk))


def _quotas(override=None, pilot=0, pilot_human=0, large=2, human=1):
    """The same quota for every stratum, then `override`'s by stratum."""
    q = {s: Quota(pilot, pilot_human, large, human) for s in STRATA}
    q.update(override or {})
    return q


def _order(seed, ids):
    """The seeded order the spec defines, written out independently of the sampler."""
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


# --- strata -----------------------------------------------------------------


@pytest.mark.parametrize(
    "purpose, risk, stratum",
    [
        (["verbatim", "spread"], ["stitched"], ("both", "rare")),
        (["verbatim"], ["invented"], ("both", "rare")),  # over both kinds together
        (["verbatim", "stitched"], [], ("purpose_only", "stitched")),
        ([], ["verbatim", "verbatim"], ("risk_only", "verbatim")),
        ([], [], ("neither", None)),
    ],
)
def test_each_label_gets_its_stratum_by_rarest_type(purpose, risk, stratum):
    pilot = draw_pilot(_profile(_label(1, purpose, risk)), _quotas(pilot=1), SEED)

    assert [(l.dsld_id, l.stratum) for l in pilot] == [(1, stratum)]
    assert pilot[0].case_id == "label:1"


# --- seeded order and nesting ----------------------------------------------


def test_the_pilot_is_the_head_of_the_seeded_order_its_first_flagged_human():
    labels = [_label(i, purpose=["verbatim"]) for i in range(1, 21)]
    quotas = _quotas({("purpose_only", "verbatim"): Quota(5, 2, 10, 3)})

    pilot = draw_pilot(_profile(*labels), quotas, SEED)

    expected = _order(SEED, range(1, 21))[:5]
    assert [l.dsld_id for l in pilot] == expected
    assert [l.rank for l in pilot] == [0, 1, 2, 3, 4]
    assert [l.in_human for l in pilot] == [True, True, False, False, False]


def test_every_set_is_a_prefix_of_one_order():
    labels = [_label(i, purpose=["verbatim"]) for i in range(1, 21)]
    quotas = _quotas({("purpose_only", "verbatim"): Quota(5, 2, 10, 3)})
    prof = _profile(*labels)

    sample = draw(prof, quotas, SEED, draw_pilot(prof, quotas, SEED))

    order = _order(SEED, range(1, 21))
    assert sample.seed == SEED
    assert [l.dsld_id for l in sample.large] == order[5:15]
    assert [l.rank for l in sample.large] == list(range(5, 15))
    assert [l.dsld_id for l in sample.large if l.in_human] == order[5:8]
    size = next(z for z in sample.sizes if z.stratum == ("purpose_only", "verbatim"))
    assert (size.in_run, size.pilot, size.pilot_human, size.large, size.human) == (20, 5, 2, 10, 3)


def test_the_run_s_row_order_does_not_matter():
    labels = [_label(i, purpose=["verbatim"]) for i in range(1, 21)]
    quotas = _quotas({("purpose_only", "verbatim"): Quota(5, 2, 10, 3)})
    forward, backward = _profile(*labels), _profile(*reversed(labels))

    a = draw(forward, quotas, SEED, draw_pilot(forward, quotas, SEED))
    b = draw(backward, quotas, SEED, draw_pilot(backward, quotas, SEED))

    assert a == b


def test_a_quota_above_what_is_left_takes_all_of_it():
    labels = [_label(i, risk=["invented"]) for i in range(1, 8)]
    quotas = _quotas({("risk_only", "rare"): Quota(5, 2, 100, 10)})
    prof = _profile(*labels)

    sample = draw(prof, quotas, SEED, draw_pilot(prof, quotas, SEED))

    size = next(z for z in sample.sizes if z.stratum == ("risk_only", "rare"))
    assert (size.in_run, size.pilot, size.large, size.human) == (7, 5, 2, 2)
    assert {l.dsld_id for l in sample.large} == set(_order(SEED, range(1, 8))[5:])


def test_every_stratum_has_a_size_row_empty_ones_with_zeros():
    prof = _profile(_label(1, purpose=["verbatim"], risk=["verbatim"]), *[_label(i) for i in range(2, 6)])

    sample = draw(prof, _quotas(), SEED, draw_pilot(prof, _quotas(), SEED))

    assert [z.stratum for z in sample.sizes] == STRATA
    assert sum(z.in_run for z in sample.sizes) == 5
    empty = next(z for z in sample.sizes if z.stratum == ("risk_only", "rare"))
    assert (empty.in_run, empty.pilot, empty.large, empty.human) == (0, 0, 0, 0)


# --- edge cases -------------------------------------------------------------


def test_a_pilot_that_does_not_match_raises():
    labels = [_label(i, purpose=["verbatim"]) for i in range(1, 21)]
    quotas = _quotas({("purpose_only", "verbatim"): Quota(5, 2, 10, 3)})
    prof = _profile(*labels)

    with pytest.raises(ValueError, match="changed since the pilot"):
        draw(prof, quotas, SEED, draw_pilot(prof, quotas, SEED + 1))


def test_a_quota_for_an_unknown_stratum_raises():
    with pytest.raises(ValueError, match="exactly the 10 strata"):
        draw_pilot(_profile(), _quotas({("neither", "verbatim"): Quota(1, 1, 1, 1)}), SEED)


def test_a_stratum_with_no_quota_raises():
    quotas = _quotas()
    del quotas[("both", "rare")]

    with pytest.raises(ValueError, match="exactly the 10 strata"):
        draw_pilot(_profile(), quotas, SEED)


@pytest.mark.parametrize("quota", [Quota(1, 2, 5, 1), Quota(1, 1, 5, 6)])
def test_a_quota_that_does_not_nest_raises(quota):
    with pytest.raises(ValueError, match="does not nest"):
        draw_pilot(_profile(), _quotas({("neither", None): quota}), SEED)


def test_a_stratum_the_pilot_uses_up_raises():
    prof = _profile(*[_label(i) for i in range(1, 4)])
    quotas = _quotas({("neither", None): Quota(5, 2, 10, 3)})

    pilot = draw_pilot(prof, quotas, SEED)
    assert len(pilot) == 3
    with pytest.raises(ValueError, match="weight is undefined"):
        draw(prof, quotas, SEED, pilot)


# --- build, save and read ---------------------------------------------------

PASSAGE = "do not take if pregnant\nsupports healthy joints\nkeep out of reach of children"

ROWS = [
    {"dsld_id": 1, "status": "ok", "purpose": [{"text": "supports healthy joints"}], "risk": [{"text": "do not take if pregnant"}]},
    {"dsld_id": 2, "status": "ok", "purpose": [{"text": "joints healthy supports"}, {"text": "healthy joints"}], "risk": []},
    {"dsld_id": 3, "status": "error", "purpose": [], "risk": []},
    {"dsld_id": 4, "status": "ok", "purpose": [], "risk": [{"text": "cures arthritis"}]},
    {"dsld_id": 5, "status": "ok", "purpose": [{"text": "healthy joints"}], "risk": [{"text": "keep out of reach"}]},
    *({"dsld_id": i, "status": "ok", "purpose": [], "risk": []} for i in (6, 7, 8)),
]

# One pilot label from `both, verbatim` (1 and 5) and from `neither` (6 to 8).
QUOTAS = _quotas({("both", "verbatim"): Quota(1, 1, 1, 1), ("neither", None): Quota(1, 0, 5, 1)})


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A profiled run `tiny-…` over label set `tiny`, label 3 failed."""
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
    yield d
    label_loader.reset()


def test_build_then_read_joins_each_label_with_its_claims(run):
    pilot = build_pilot(run, QUOTAS, SEED)
    sample = build_evaluation_set(run, QUOTAS, SEED)

    assert [c.label for c in runs.read_pilot(run)] == pilot
    cases = runs.read_evaluation_set(run)
    assert [c.label for c in cases] == sample.large
    by_id = {c.label.dsld_id: c for c in cases}
    assert set(by_id) == {2, 4, *({1, 5} - {pilot[0].dsld_id}), *({6, 7, 8} - {pilot[1].dsld_id})}
    two = by_id[2]
    assert two.label.stratum == ("purpose_only", "stitched")
    assert [(c.claim_id, c.text, c.type) for c in two.purpose] == [
        ("claim:2:purpose:0", "joints healthy supports", "stitched"),
        ("claim:2:purpose:1", "healthy joints", "verbatim"),
    ]
    assert two.risk == []
    assert [(c.claim_id, c.type) for c in by_id[4].risk] == [("claim:4:risk:0", "invented")]


def test_the_manifests_pin_the_profile_the_rows_and_the_pilot(run):
    build_pilot(run, QUOTAS, SEED)
    build_evaluation_set(run, QUOTAS, SEED)

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    pm = json.loads(run_paths.eva_manifest_path(run, run_paths.PILOT_MANIFEST_NAME).read_text())
    em = json.loads(run_paths.eva_manifest_path(run, run_paths.EVALUATION_SET_MANIFEST_NAME).read_text())
    for m in (pm, em):
        assert m["run"] == run.name
        assert m["seed"] == SEED
        assert m["profile_sha"] == sha(run_paths.rows_path(run, run_paths.PROFILE_NAME))
        assert m["rows_sha"] == sha(run_paths.rows_path(run))
        assert m["failed"] == 1
        assert [(q["group"], q["rarest"]) for q in m["quotas"]] == STRATA
    assert em["pilot_sha"] == sha(run_paths.eva_path(run, run_paths.PILOT_NAME))
    neither = next(z for z in em["sizes"] if z["group"] == "neither")
    assert neither == {"group": "neither", "rarest": None, "in_run": 3, "pilot": 1, "pilot_human": 0, "large": 2, "human": 1}
    stored = json.loads(run_paths.eva_path(run, run_paths.PILOT_NAME).read_text().splitlines()[0])
    assert set(stored) == {"case_id", "dsld_id", "group", "rarest", "rank", "in_human"}


def test_a_second_save_refuses(run):
    build_pilot(run, QUOTAS, SEED)
    build_evaluation_set(run, QUOTAS, SEED)

    with pytest.raises(FileExistsError):
        build_pilot(run, QUOTAS, SEED)
    with pytest.raises(FileExistsError):
        build_evaluation_set(run, QUOTAS, SEED)


def test_a_changed_profile_makes_the_read_raise(run):
    build_pilot(run, QUOTAS, SEED)
    p = run_paths.rows_path(run, run_paths.PROFILE_NAME)
    p.write_text(p.read_text() + "\n")

    with pytest.raises(ValueError, match="stale"):
        runs.read_pilot(run)


def test_changed_rows_make_the_read_raise(run):
    build_pilot(run, QUOTAS, SEED)
    build_evaluation_set(run, QUOTAS, SEED)
    p = run_paths.rows_path(run)
    p.write_text(p.read_text() + "\n")

    with pytest.raises(ValueError, match="stale"):
        runs.read_evaluation_set(run)


def test_the_evaluation_set_before_the_pilot_raises(run):
    with pytest.raises(FileNotFoundError, match="pilot"):
        build_evaluation_set(run, QUOTAS, SEED)


def test_a_profile_of_another_classifier_version_raises(run):
    p = run_paths.manifest_path(run, run_paths.PROFILE_MANIFEST_NAME)
    p.write_text(json.dumps({**json.loads(p.read_text()), "classifier_version": -1}))

    with pytest.raises(ValueError, match="classifier_version"):
        build_pilot(run, QUOTAS, SEED)


def test_a_run_without_a_profile_raises(run):
    run_paths.manifest_path(run, run_paths.PROFILE_MANIFEST_NAME).unlink()
    run_paths.rows_path(run, run_paths.PROFILE_NAME).unlink()

    with pytest.raises(FileNotFoundError, match="no profile"):
        build_pilot(run, QUOTAS, SEED)


def test_a_stale_profile_raises(run):
    p = run_paths.rows_path(run)
    p.write_text(p.read_text() + "\n")

    with pytest.raises(ValueError, match="stale"):
        build_pilot(run, QUOTAS, SEED)
