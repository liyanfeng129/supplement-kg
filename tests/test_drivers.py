"""Three drivers, one row contract.

The enforced abstraction is the output row plus `validate()`, not a Backend
class -- so the thing worth testing is that a row written by a driver that
never touches a GPU is indistinguishable in shape, identity and grounding from
one the local driver wrote. `provenance` is a column, never a branch.

The agent driver is the one that cannot be driven from a script (subagents are
spawned inside an agent's turn), so its two halves are exercised directly:
`start_agent_run` opens the run, `append_agent_row` lands each result.
"""

from __future__ import annotations

import json

import pytest

from engine.pipeline import store
from engine.pipeline.config import Config
from engine.pipeline.drivers import append_agent_row, start_agent_run

PASSAGE = "PRODUCT: Thing\n\nLABEL STATEMENTS:\n[Precautions] KEEP OUT OF REACH OF CHILDREN."


@pytest.fixture
def records(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"dsld_id": 709, "name": "Thing", "passage": PASSAGE}) + "\n"
    )
    return path


def _raw(evidence="KEEP OUT OF REACH OF CHILDREN."):
    return {
        "purpose": [],
        "risk": [{"text": "keep out of reach of children", "evidence": evidence}],
        "purpose_confidence": 0,
        "risk_confidence": 3,
    }


def test_an_agent_row_carries_the_same_identity_as_a_local_row(tmp_path, records):
    """Rows from different drivers must pool: same config, same hash."""
    cfg = Config(provenance="agent")
    d, recs = start_agent_run(records, config=cfg, runs_root=tmp_path / "runs")

    row = append_agent_row(d, _raw(), recs[0]["passage"], dsld_id=709, config=cfg)

    assert row["config_hash"] == Config().config_hash
    assert row["provenance"] == "agent"
    assert row["status"] == "ok"


def test_latency_is_null_for_the_agent_driver(tmp_path, records):
    """A subagent's wall-clock is not comparable to inference latency."""
    cfg = Config(provenance="agent")
    d, recs = start_agent_run(records, config=cfg, runs_root=tmp_path / "runs")

    row = append_agent_row(d, _raw(), recs[0]["passage"], dsld_id=709, config=cfg)

    assert row["latency_ms"] is None


def test_an_agent_row_is_grounded_by_the_same_shared_check(tmp_path, records):
    """Grounding is shared by all three drivers, not reimplemented per driver."""
    cfg = Config(provenance="agent")
    d, recs = start_agent_run(records, config=cfg, runs_root=tmp_path / "runs")

    good = append_agent_row(d, _raw(), recs[0]["passage"], dsld_id=709, config=cfg)
    bad = append_agent_row(
        d, _raw("Clinically proven to cure insomnia"), recs[0]["passage"],
        dsld_id=710, config=cfg,
    )

    assert good["risk"][0]["grounded"] is True
    assert bad["risk"][0]["grounded"] is False
    # The confidence cap applies to an agent row exactly as to a local one.
    assert bad["risk_confidence"] == 1


def test_agent_rows_land_in_the_run_directory_as_jsonl(tmp_path, records):
    cfg = Config(provenance="agent")
    d, recs = start_agent_run(records, config=cfg, runs_root=tmp_path / "runs")
    append_agent_row(d, _raw(), recs[0]["passage"], dsld_id=709, config=cfg)

    assert [r["dsld_id"] for r in store.read_rows(d)] == [709]
    assert store.read_manifest(d)["provenance"] == "agent"


def test_the_agent_run_records_the_prompt_verbatim_like_any_other(tmp_path, records):
    cfg = Config(provenance="agent", note="hand-directed pass")
    d, _ = start_agent_run(records, config=cfg, runs_root=tmp_path / "runs")
    manifest = store.read_manifest(d)

    assert manifest["prompt"] == Config().prompt
    assert manifest["note"] == "hand-directed pass"
