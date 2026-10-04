"""PurposeRiskService: loader -> controller -> store, end to end on a fake model.

What is pinned (spec 0003, Acceptance): a run over N labels writes N rows, one
per id, with the status each outcome earned; `ok` rows carry the parsed
purpose and risk, failed rows empty lists and the raw text; a request that
keeps failing is retried, then written as `error`, and the run goes on; a
killed run resumed writes every id exactly once, and a resume under a changed
prompt is refused; `ModelUnavailable` stops the run with every finished row on
disk; crossing the failure threshold notifies exactly once.

No GPU: the controller drives a stand-in model that answers by passage. The
run directory and the label file live in `tmp_path`; `notify` is recorded.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from engine import config
from engine.controllers.qwen_controller import QwenController
from engine.models.base_model import BaseModel, Generation, ModelUnavailable, Prompt, Usage
from engine.repositories.dsld import store
from engine.repositories.prompts import loader as prompts
from engine.services.purpose_risk import service as service_module
from engine.services.purpose_risk.service import PurposeRiskService

GOOD = "P: supports restful sleep | 4\nR: may cause drowsiness | 3"


class ByPassage(BaseModel):
    """Answers from the passage's first word; records what it saw.

    `good` answers well; `bad` answers in a format every check rejects;
    `boom` fails the request; `gone` says the model cannot run; `long` does
    not fit. `kill_after` interrupts the process on that many-plus-one-th
    request, as Ctrl-C or a kill would.
    """

    def __init__(self, kill_after: int | None = None):
        self.kill_after = kill_after
        self.requests = 0
        self.closed = False

    def fits(self, prompt: Prompt) -> bool:
        return not prompt.user.startswith("long")

    def generate(self, prompt: Prompt) -> Generation:
        self.requests += 1
        if self.kill_after is not None and self.requests > self.kill_after:
            raise KeyboardInterrupt
        word = prompt.user.split()[0]
        if word == "boom":
            raise RuntimeError("connection reset by peer")
        if word == "gone":
            raise ModelUnavailable("llama-server exited with code -11")
        text = "Here are the claims" if word == "bad" else GOOD
        return Generation(
            text=text,
            latency_ms=1200,
            usage=Usage(prompt_tokens=50, completion_tokens=9),
            stop_reason="eos",
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Runs under tmp_path, notify recorded, the committed engine.toml thresholds."""
    monkeypatch.setattr(store, "RUNS_ROOT", tmp_path / "runs")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(service_module, "notify", lambda subject, body: sent.append((subject, body)))
    return {"tmp": tmp_path, "sent": sent}


def labels_file(tmp_path, passages: list[str]):
    """One label per passage, ids 1..N, as the pilot writes them."""
    path = tmp_path / "passages.jsonl"
    with path.open("w") as fh:
        for i, passage in enumerate(passages, 1):
            fh.write(json.dumps({"dsld_id": i, "name": f"label {i}", "passage": passage}) + "\n")
    return path


def service(model: BaseModel, **kw) -> PurposeRiskService:
    return PurposeRiskService(QwenController(model, max_attempts=2, in_flight=kw.pop("in_flight", 2)))


def the_run(env):
    [run] = sorted((env["tmp"] / "runs").iterdir())
    return run


def rows_by_id(run) -> dict[int, dict]:
    rows = store.read_rows(run)
    by_id = {r["dsld_id"]: r for r in rows}
    assert len(by_id) == len(rows), "an id was written twice"
    return by_id


# --- one row per label ------------------------------------------------------


def test_a_run_writes_one_row_per_label_with_the_status_it_earned(env):
    path = labels_file(env["tmp"], ["good one", "bad two", "boom three", "long four", "good five"])

    with service(ByPassage()) as svc:
        svc.run(path=path)

    rows = rows_by_id(the_run(env))
    assert {i: r["status"] for i, r in rows.items()} == {
        1: "ok", 2: "failed_checks", 3: "error", 4: "too_long", 5: "ok",
    }


def test_ok_rows_carry_the_parsed_terms_and_failed_rows_the_raw_text(env):
    path = labels_file(env["tmp"], ["good one", "bad two"])

    with service(ByPassage()) as svc:
        svc.run(path=path)

    rows = rows_by_id(the_run(env))
    ok, failed = rows[1], rows[2]
    assert ok["purpose"] == [{"text": "supports restful sleep", "score": 4}]
    assert ok["risk"] == [{"text": "may cause drowsiness", "score": 3}]
    assert ok["text"] == GOOD
    assert (ok["attempts"], ok["failed"]) == (1, [])
    assert (ok["prompt_tokens"], ok["completion_tokens"], ok["latency_ms"]) == (50, 9, 1200)
    assert (failed["purpose"], failed["risk"]) == ([], [])
    assert failed["text"] == "Here are the claims"
    assert failed["attempts"] == 2 and failed["failed"]


def test_a_failing_request_is_retried_then_written_as_error_and_the_run_goes_on(env):
    path = labels_file(env["tmp"], ["boom one", "good two"])
    model = ByPassage()

    with service(model, in_flight=1) as svc:
        svc.run(path=path)

    rows = rows_by_id(the_run(env))
    assert (rows[1]["status"], rows[1]["attempts"], rows[1]["text"]) == ("error", 2, "")
    assert rows[2]["status"] == "ok"
    assert model.requests == 3


def test_every_row_and_the_manifest_carry_one_config_hash(env):
    path = labels_file(env["tmp"], ["good one", "good two"])

    with service(ByPassage()) as svc:
        svc.run(path=path, limit=2, order="sorted")

    run = the_run(env)
    manifest = store.read_manifest(run)
    assert manifest["selection"] == {
        "path": str(path), "ids": None, "limit": 2, "order": "sorted", "seed": None,
    }
    assert {r["config_hash"] for r in store.read_rows(run)} == {manifest["config_hash"]}


def test_the_run_finishes_with_one_notice_and_one_printed_line(env, capsys):
    path = labels_file(env["tmp"], ["good one", "bad two"])

    with service(ByPassage()) as svc:
        svc.run(path=path)

    [(subject, body)] = env["sent"]
    assert "finished" in subject
    assert "ok 1" in body and "failed_checks 1" in body
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "ok 1" in out[0]


def test_leaving_the_block_closes_the_controller_and_its_model(env):
    model = ByPassage()

    with service(model):
        pass

    assert model.closed


# --- resume -----------------------------------------------------------------


def test_a_killed_run_resumed_writes_every_id_exactly_once(env):
    path = labels_file(env["tmp"], [f"good {i}" for i in range(1, 9)])

    with pytest.raises(KeyboardInterrupt):
        with service(ByPassage(kill_after=3), in_flight=1) as svc:
            svc.run(path=path, limit=6)
    run = the_run(env)
    assert set(rows_by_id(run)) == {1, 2, 3}

    with service(ByPassage()) as svc:
        svc.run(run=run)

    assert set(rows_by_id(run)) == {1, 2, 3, 4, 5, 6}  # the manifest's limit, not 6 more


def test_a_resume_under_a_changed_prompt_is_refused(env, monkeypatch):
    path = labels_file(env["tmp"], ["good one", "good two"])
    with service(ByPassage()) as svc:
        svc.run(path=path, limit=1)
    run = the_run(env)

    real = prompts.load
    monkeypatch.setattr(prompts, "load", lambda pid: replace(real(pid), system="a new prompt"))

    with pytest.raises(ValueError, match="config_hash"):
        with service(ByPassage()) as svc:
            svc.run(run=run)
    assert set(rows_by_id(run)) == {1}


def test_a_resume_takes_its_selection_from_the_manifest_not_the_call(env):
    path = labels_file(env["tmp"], ["good one"])
    with service(ByPassage()) as svc:
        svc.run(path=path)

    with pytest.raises(ValueError, match="manifest"):
        with service(ByPassage()) as svc:
            svc.run(run=the_run(env), limit=5)


# --- stopping and warning ---------------------------------------------------


def test_model_unavailable_stops_the_run_with_every_finished_row_on_disk(env):
    path = labels_file(env["tmp"], ["good one", "good two", "gone three", "good four"])

    with pytest.raises(ModelUnavailable, match="exited"):
        with service(ByPassage(), in_flight=1) as svc:
            svc.run(path=path)

    assert set(rows_by_id(the_run(env))) == {1, 2}
    [(subject, body)] = env["sent"]
    assert "unavailable" in subject and "exited" in body


def test_crossing_the_failure_threshold_notifies_exactly_once(env, monkeypatch):
    engine_toml = env["tmp"] / "engine.toml"
    engine_toml.write_text("[notify]\nfail_rate = 0.2\nfail_min_labels = 4\n")
    monkeypatch.setattr(config, "ENGINE_CONFIG_PATH", engine_toml)
    path = labels_file(env["tmp"], ["bad"] * 6 + ["good"] * 2)

    with service(ByPassage(), in_flight=1) as svc:
        svc.run(path=path)

    subjects = [s for s, _ in env["sent"]]
    assert sum("fail" in s for s in subjects) == 1
    assert len(subjects) == 2  # the warning, then the finish
    assert len(store.read_rows(the_run(env))) == 8  # warned, did not stop


def test_below_the_minimum_no_warning_is_sent(env, monkeypatch):
    engine_toml = env["tmp"] / "engine.toml"
    engine_toml.write_text("[notify]\nfail_rate = 0.2\nfail_min_labels = 10\n")
    monkeypatch.setattr(config, "ENGINE_CONFIG_PATH", engine_toml)
    path = labels_file(env["tmp"], ["bad"] * 5)

    with service(ByPassage()) as svc:
        svc.run(path=path)

    assert [s for s, _ in env["sent"] if "fail" in s] == []
