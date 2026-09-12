"""One pathological label must not stall a multi-day job.

The run loop is not a fixture seam -- it is the GPU and IO path, covered by
`smoke` -- but the poison guard is the one part of it whose failure is silent
and expensive: a record that always fails is handed back forever, and a
multi-day corpus run never finishes. That is cheap to test with a fake
extractor and impossible to notice otherwise.
"""

from __future__ import annotations

import json

import pytest

from engine.pipeline import run as run_mod
from engine.pipeline import store
from engine.pipeline.config import Config


class _FakeExtractor:
    """Stands in for the model: fails for `bad_ids`, succeeds otherwise."""

    def __init__(self, config, bad_ids=(), passage_to_id=None):
        self.config = config
        self.bad = set(bad_ids)
        self.passage_to_id = passage_to_id or {}
        self.calls = 0

    def extract(self, passage: str):
        self.calls += 1
        if self.passage_to_id.get(passage) in self.bad:
            raise RuntimeError("pathological label")
        return (
            json.dumps(
                {
                    "purpose": [{"text": "supports sleep", "evidence": "Supports Sleep"}],
                    "risk": [],
                    "purpose_confidence": 4,
                    "risk_confidence": 0,
                }
            ),
            1234,
        )


@pytest.fixture
def records(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        {"dsld_id": 1, "name": "Good", "passage": "PRODUCT: Good\nSupports Sleep"},
        {"dsld_id": 2, "name": "Bad", "passage": "PRODUCT: Bad\nSupports Sleep"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path, {r["passage"]: r["dsld_id"] for r in rows}


def _run(records, tmp_path, monkeypatch, bad_ids=()):
    path, p2i = records
    fake = _FakeExtractor(Config(), bad_ids=bad_ids, passage_to_id=p2i)
    monkeypatch.setattr(run_mod, "LocalExtractor", lambda cfg: fake)
    d = run_mod.run(
        path,
        config=Config(),
        runs_root=tmp_path / "runs",
        skip_smoke=True,
        progress_interval=10_000,
    )
    return d, fake


def test_a_record_that_always_fails_is_written_off_as_poison(tmp_path, records, monkeypatch):
    d, _ = _run(records, tmp_path, monkeypatch, bad_ids={2})
    by_id = {r["dsld_id"]: r for r in store.read_rows(d)}

    assert by_id[2]["status"] == "poison"
    assert by_id[1]["status"] == "ok"


def test_the_poison_guard_bounds_how_often_one_record_is_retried(tmp_path, records, monkeypatch):
    """Unbounded retries are how one label stalls a 14-day run."""
    _, fake = _run(records, tmp_path, monkeypatch, bad_ids={2})

    # One successful call plus at most MAX_ATTEMPTS on the failing record.
    assert fake.calls <= 1 + run_mod.MAX_ATTEMPTS


def test_every_record_produces_exactly_one_row(tmp_path, records, monkeypatch):
    """Re-queuing must not duplicate rows or lose them."""
    d, _ = _run(records, tmp_path, monkeypatch, bad_ids={2})
    rows = store.read_rows(d)

    assert len(rows) == 2
    assert sorted(r["dsld_id"] for r in rows) == [1, 2]


def test_a_clean_run_writes_one_ok_row_per_record(tmp_path, records, monkeypatch):
    d, _ = _run(records, tmp_path, monkeypatch)
    rows = store.read_rows(d)

    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert all(r["provenance"] == "local" for r in rows)
    assert all(r["config_hash"] == Config().config_hash for r in rows)


def test_resume_skips_records_already_done(tmp_path, records, monkeypatch):
    """Resume is input IDs minus IDs present in the output -- nothing else."""
    d, _ = _run(records, tmp_path, monkeypatch)

    path, p2i = records
    fake = _FakeExtractor(Config(), passage_to_id=p2i)
    monkeypatch.setattr(run_mod, "LocalExtractor", lambda cfg: fake)
    run_mod.run(
        path,
        config=Config(),
        runs_root=tmp_path / "runs",
        resume=d.name,
        skip_smoke=True,
        progress_interval=10_000,
    )

    assert fake.calls == 0
    assert len(store.read_rows(d)) == 2


def test_the_manifest_records_the_grounding_rate_and_statuses(tmp_path, records, monkeypatch):
    d, _ = _run(records, tmp_path, monkeypatch)
    manifest = store.read_manifest(d)

    assert manifest["statuses"] == {"ok": 2}
    assert manifest["grounding_rate"] == 1.0
    assert manifest["n_written"] == 2


def test_retry_targets_only_the_rows_that_are_not_ok(tmp_path, records, monkeypatch):
    """`empty` is a success and must never be retried."""
    d, _ = _run(records, tmp_path, monkeypatch, bad_ids={2})

    assert run_mod.retry_ids(d) == [2]


def test_a_run_with_nothing_to_retry_reports_so_and_writes_no_run_dir(tmp_path, records, monkeypatch):
    d, _ = _run(records, tmp_path, monkeypatch)
    path, _p2i = records

    assert run_mod.retry_ids(d) == []
    assert run_mod.retry(d, path, config=Config(), runs_root=tmp_path / "runs") is None


def test_a_retry_lands_in_its_own_run_directory(tmp_path, records, monkeypatch):
    """Retried work is attributable, never blended into the original run."""
    d, _ = _run(records, tmp_path, monkeypatch, bad_ids={2})
    path, p2i = records

    fake = _FakeExtractor(Config(), passage_to_id=p2i)  # now succeeds
    monkeypatch.setattr(run_mod, "LocalExtractor", lambda cfg: fake)
    retried = run_mod.retry(
        d, path, config=Config(), runs_root=tmp_path / "runs", skip_smoke=True,
        progress_interval=10_000,
    )

    assert retried != d
    rows = store.read_rows(retried)
    assert [r["dsld_id"] for r in rows] == [2]
    assert rows[0]["status"] == "ok"
    # The original run keeps its failure, so its failure rate stays readable.
    assert {r["dsld_id"]: r["status"] for r in store.read_rows(d)}[2] == "poison"
