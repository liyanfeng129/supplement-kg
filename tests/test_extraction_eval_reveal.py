"""The reveal (0011.03 Q15): items where the human and the judges differ, and checked, appended decisions.

What is pinned: only saved labels are shown; overlapping missed quotes share one
item with the human's blind value; a claim on which all three give the same verdict
is not an item; `save` stores nothing and names the problem on a value off the
item's list or an unknown item; the last decision on an item is the one read back,
a torn last line is skipped; and the blind verdicts file is never touched.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from engine.repositories.dsld.runs import loader as runs
from engine.repositories.dsld.runs import paths as run_paths
from engine.repositories.dsld.runs import store as run_store
from engine.repositories.dsld.runs.schema import ClaimVerdict, HumanVerdict, MissedClaim
from engine.services.extraction_eval.reveal import reveal_labels, save
from engine.services.extraction_eval.schema import SheetCase, SheetClaim
from web.reveal import server as reveal_server

PASSAGE = "supports healthy joints and eases stiffness\ndo not take if pregnant"
CASE = SheetCase("label:1", 1, PASSAGE, [
    SheetClaim("claim:1:purpose:0", "purpose", "supports healthy joints"),
    SheetClaim("claim:1:risk:0", "risk", "do not take if pregnant"),
])
OTHER = SheetCase("label:2", 2, "nothing here", [])


def judge(first: dict) -> dict:
    """A judge's stored verdict on label 1: `first` over a correct first claim, the risk claim correct."""
    return {
        "case_id": "label:1", "prompt_sha": "abc", "missed": [],
        "claims": [
            {"claim_id": "claim:1:purpose:0", "faithful": "yes", "actual_kind": "purpose",
             "confidence": "high", "rule": "", "evidence": "", "reason": "", **first},
            {"claim_id": "claim:1:risk:0", "faithful": "yes", "actual_kind": "risk",
             "confidence": "high", "rule": "", "evidence": "", "reason": ""},
        ],
    }


@pytest.fixture
def run(tmp_path):
    """A run whose human saved label 1, with a missed quote and a correct verdict on both claims."""
    run_store.save_human_verdict(tmp_path, HumanVerdict(
        "label:1", 1,
        [ClaimVerdict("claim:1:purpose:0", "yes", "purpose", False, "my note"),
         ClaimVerdict("claim:1:risk:0", "yes", "risk", False, "")],
        [MissedClaim("eases stiffness", "purpose", False, "")],
        "t0", "t1", 5,
    ))
    return tmp_path


@pytest.fixture
def judged():
    claude = judge({"faithful": "meaning_changed", "rule": "D0", "reason": "drops a condition", "evidence": "supports"})
    claude["missed"] = [{"quote": "and eases stiffness", "kind": "purpose", "rule": "def.purpose", "confidence": "low"}]
    codex = judge({})
    codex["missed"] = [{"quote": "healthy joints and", "kind": "risk", "rule": "def.risk", "confidence": "high"}]
    return {"claude": {"label:1": claude}, "codex": {"label:1": codex}}


def test_items_are_the_differences_on_saved_labels(run, judged):
    labels = reveal_labels(run, [CASE, OTHER], judged)
    assert [x.case_id for x in labels] == ["label:1"]
    missed, *rest = [x for x in labels[0].items if x.item_kind == "missed"]
    assert missed.item == "and eases stiffness | eases stiffness"
    assert missed.blind == "purpose" and missed.quotes == ["eases stiffness"]
    assert [o.quote for o in missed.opinions["claude"]] == ["and eases stiffness"] and missed.opinions["codex"] == []
    assert [x.item for x in rest] == ["healthy joints and"]
    assert rest[0].blind == "none"

    claims = [x for x in labels[0].items if x.item_kind == "claim"]
    assert [x.item for x in claims] == ["claim:1:purpose:0"]  # the risk claim: all three agree
    assert claims[0].blind == "yes/purpose" and claims[0].comment == "my note"
    assert claims[0].opinions["claude"][0].value == "meaning_changed/purpose"
    assert claims[0].opinions["claude"][0].reason == "drops a condition"
    assert labels[0].prompt_sha == "abc"


@pytest.mark.parametrize("item,final", [
    ("claim:1:purpose:0", "yes"),
    ("claim:1:purpose:0", "none"),
    ("claim:1:purpose:0", "invented/purpose"),
    ("eases stiffness | and eases stiffness", "none"),  # not the item's id
    ("and eases stiffness | eases stiffness", "yes/purpose"),
    ("and eases stiffness | eases stiffness", None),
])
def test_save_refuses_and_stores_nothing(run, judged, item, final):
    label = reveal_labels(run, [CASE], judged)[0]
    assert save(run, label, item, final, "")
    assert runs.read_adjudications(run) == {}


def test_last_decision_counts_and_a_torn_tail_is_skipped(run, judged):
    label = reveal_labels(run, [CASE], judged)[0]
    human_before = run_paths.eva_path(run, run_paths.HUMAN_PILOT_NAME).read_bytes()
    assert save(run, label, "healthy joints and", "none", "") == []
    assert save(run, label, "healthy joints and", "risk", "rules unclear") == []
    assert save(run, label, "claim:1:purpose:0", "invented", "") == []

    path = run_paths.eva_path(run, run_paths.ADJUDICATION_PILOT_NAME)
    with path.open("a") as fh:
        fh.write('{"case_id": "label:1", "ite')  # torn write
    read = runs.read_adjudications(run)
    assert set(read) == {("label:1", "healthy joints and"), ("label:1", "claim:1:purpose:0")}
    a = read[("label:1", "healthy joints and")]
    assert (a.item_kind, a.blind, a.final, a.gap, a.prompt_sha) == ("missed", "none", "risk", "rules unclear", "abc")

    # The next save mends the torn tail rather than gluing onto it.
    assert save(run, label, "claim:1:purpose:0", "yes/purpose", "") == []
    assert runs.read_adjudications(run)[("label:1", "claim:1:purpose:0")].final == "yes/purpose"
    assert run_paths.eva_path(run, run_paths.HUMAN_PILOT_NAME).read_bytes() == human_before


def test_server_serves_labels_and_stores_a_decision(run, judged):
    server = reveal_server.start(run, [CASE], judged, port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"<title>Reveal</title>" in urllib.request.urlopen(url + "/").read()
        data = json.load(urllib.request.urlopen(url + "/api/labels"))
        assert data["judges"] == ["claude", "codex"] and data["decided"] == []
        assert data["labels"][0]["items"][0]["item"] == "and eases stiffness | eases stiffness"

        def post(final: str) -> dict:
            body = json.dumps({"case_id": "label:1", "item": "healthy joints and", "final": final, "gap": "n"}).encode()
            return json.load(urllib.request.urlopen(urllib.request.Request(url + "/api/decision", body)))

        assert post("nope")["problems"]
        assert post("none")["problems"] == []
        data = json.load(urllib.request.urlopen(url + "/api/labels"))
        assert [(a["item"], a["final"], a["gap"]) for a in data["decided"]] == [("healthy joints and", "none", "n")]
    finally:
        server.shutdown()
        server.server_close()
