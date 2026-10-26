"""The profile: each claim's type, each label's group, the counts (spec 0011.01).

What is pinned: the four types, first match wins, against the passage the
model read -- a substring is verbatim, words from one line in any order are
stitched, words from several lines are spread, any foreign word is invented.
A label's group comes from its row alone. A failed row is only listed; an
unknown id, a second row for an id and an empty claim raise. Every cell of the
count cube is present. `build_profile` saves what `profile` computes, refuses
a run that is already profiled before scanning it, and refuses a label set the
run did not read.
"""

from __future__ import annotations

import json

import pytest

from engine.repositories.dsld.labels import loader as label_loader
from engine.repositories.dsld.labels import paths as label_paths
from engine.repositories.dsld.labels import store as label_store
from engine.repositories.dsld.labels.schema import Label
from engine.repositories.dsld.runs import loader as runs
from engine.repositories.dsld.runs import paths as run_paths
from engine.repositories.dsld.runs import store as run_store
from engine.services.extraction_eval import profile as prof
from engine.services.extraction_eval.profile import build_profile, profile

PASSAGE = "do not take if pregnant\nsupports healthy joints\nkeep out of reach of children"


def _row(dsld_id, purpose=(), risk=(), status="ok"):
    return {
        "dsld_id": dsld_id,
        "status": status,
        "purpose": [{"text": t} for t in purpose],
        "risk": [{"text": t} for t in risk],
    }


def _labels(*ids, passage=PASSAGE):
    return {i: Label(i, f"label {i}", passage) for i in ids}


# --- types ------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim, expected",
    [
        ("supports healthy joints", "verbatim"),
        ("healthy joints", "verbatim"),
        ("joints healthy supports", "stitched"),  # one line, another order
        ("do not take if children", "spread"),  # every word, but over two lines
        ("cures arthritis", "invented"),
    ],
)
def test_each_claim_gets_its_type(claim, expected):
    p = profile([_row(1, purpose=[claim])], _labels(1))

    assert p.labels[0].purpose_types == [expected]


def test_types_follow_each_list_s_order():
    p = profile(
        [_row(1, purpose=["cures arthritis", "supports healthy joints"], risk=["do not take if pregnant"])],
        _labels(1),
    )

    assert p.labels[0].purpose_types == ["invented", "verbatim"]
    assert p.labels[0].risk_types == ["verbatim"]


# --- groups -----------------------------------------------------------------


@pytest.mark.parametrize(
    "purpose, risk, group",
    [
        (["supports healthy joints"], ["do not take if pregnant"], "both"),
        (["supports healthy joints"], [], "purpose_only"),
        ([], ["do not take if pregnant"], "risk_only"),
        ([], [], "neither"),
        (["cures arthritis"], [], "purpose_only"),  # from the row alone, whatever the type
    ],
)
def test_each_label_gets_its_group(purpose, risk, group):
    p = profile([_row(1, purpose, risk)], _labels(1))

    assert p.labels[0].group == group


# --- counts -----------------------------------------------------------------


def test_every_cell_of_the_cube_is_present_and_counted():
    p = profile(
        [
            _row(1, purpose=["supports healthy joints", "cures arthritis"], risk=["do not take if pregnant"]),
            _row(2, purpose=["joints healthy supports"]),
            _row(3),
        ],
        _labels(1, 2, 3),
    )

    assert len(p.claim_counts) == 4 * 2 * 4
    assert sum(p.claim_counts.values()) == 4
    assert p.claim_counts[("both", "purpose", "verbatim")] == 1
    assert p.claim_counts[("both", "purpose", "invented")] == 1
    assert p.claim_counts[("both", "risk", "verbatim")] == 1
    assert p.claim_counts[("purpose_only", "purpose", "stitched")] == 1
    assert p.label_counts == {"both": 1, "purpose_only": 1, "risk_only": 0, "neither": 1}


# --- edge cases -------------------------------------------------------------


def test_a_failed_row_is_only_listed():
    p = profile([_row(1, status="error"), _row(2, purpose=["supports healthy joints"])], _labels(1, 2))

    assert p.failed_ids == [1]
    assert [l.dsld_id for l in p.labels] == [2]
    assert sum(p.label_counts.values()) == 1


def test_an_id_outside_the_label_set_raises():
    with pytest.raises(ValueError, match="wrong label set"):
        profile([_row(9)], _labels(1))


def test_two_rows_for_one_id_raise():
    with pytest.raises(ValueError, match="two rows"):
        profile([_row(1), _row(1)], _labels(1))


@pytest.mark.parametrize("text", ["", "   "])
def test_an_empty_claim_raises(text):
    with pytest.raises(ValueError, match="empty"):
        profile([_row(1, risk=[text])], _labels(1))


# --- build_profile ----------------------------------------------------------


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A run `tiny-…` over label set `tiny` (labels 1 to 3), with its rows written."""
    monkeypatch.setattr(label_paths, "LABELS_ROOT", tmp_path / "labels")
    monkeypatch.setattr(run_paths, "RUNS_ROOT", tmp_path / "runs")
    label_loader.reset()
    sha = label_store.write_label_set(
        "tiny", [{"dsld_id": i, "name": f"label {i}", "passage": PASSAGE} for i in (1, 2, 3)], {}
    )
    d = run_store.new_run(
        "tiny", [1, 2, 3], {"label_set": "tiny", "passages_sha": sha, "config_hash": "abc"}
    )
    with run_paths.rows_path(d).open("w") as fh:
        for r in (
            _row(1, purpose=["supports healthy joints"], risk=["do not take if pregnant"]),
            _row(2, status="error"),
            _row(3),
        ):
            fh.write(json.dumps(r) + "\n")
    yield d
    label_loader.reset()


def test_build_profile_saves_what_profile_computes(run):
    p = build_profile(run)

    manifest, labels = runs.read_profile(run)
    assert labels == p.labels
    assert manifest["run"] == run.name
    assert manifest["config_hash"] == "abc"
    assert manifest["classifier_version"] == prof.VERSION
    assert manifest["failed_ids"] == [2]
    assert manifest["label_counts"] == p.label_counts
    assert manifest["claim_counts"]["both"]["risk"]["verbatim"] == 1


def test_a_profiled_run_is_refused_before_it_is_scanned(run, monkeypatch):
    build_profile(run)

    def scanned(*_):
        raise AssertionError("the rows were read")

    monkeypatch.setattr(runs, "read_rows", scanned)
    with pytest.raises(FileExistsError, match="already profiled"):
        build_profile(run)


def test_a_label_set_the_run_did_not_read_is_refused(run):
    m = json.loads(run_paths.manifest_path(run).read_text())
    m["passages_sha"] = "0" * 16
    run_paths.manifest_path(run).write_text(json.dumps(m))

    with pytest.raises(ValueError, match="wrong label set"):
        build_profile(run)
    assert not run_paths.manifest_path(run, run_paths.PROFILE_MANIFEST_NAME).exists()
