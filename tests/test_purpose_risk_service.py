"""PurposeRiskService: loader -> controller -> store, end to end on a fake model.

What is pinned (spec 0003, Acceptance): a run over N labels writes N rows, one
per id, with the status each outcome earned; `ok` rows carry the parsed
purpose and risk, failed rows empty lists and the raw text; a request that
keeps failing is retried, then written as `error`, and the run goes on; a
killed run resumed writes every id exactly once, and a resume under a changed
prompt is refused; `ModelUnavailable` stops the run with every finished row on
disk; crossing the failure threshold notifies exactly once.

No GPU: the controller drives a stand-in model that answers by passage. The
runs and the label sets live in `tmp_path`; `notify` is recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from engine import config
from engine.controllers.qwen_controller import QwenController
from engine.models.base_model import BaseModel, Generation, ModelUnavailable, Prompt, Usage
from engine.repositories.dsld import loader, store
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
    """Runs and label sets under tmp_path, notify recorded, the committed engine.toml thresholds."""
    monkeypatch.setattr(store, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(store, "LABELS_ROOT", tmp_path / "labels")
    monkeypatch.setattr(loader, "LABELS_ROOT", tmp_path / "labels")
    loader.reset()
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(service_module, "notify", lambda subject, body: sent.append((subject, body)))
    return {"tmp": tmp_path, "sent": sent}


def label_set(passages: list[str], name: str = "test-set") -> str:
    """One label per passage, ids 1..N, saved as a label set; returns its name."""
    rows = [{"dsld_id": i, "name": f"label {i}", "passage": p} for i, p in enumerate(passages, 1)]
    store.write_label_set(name, rows, {})
    return name


def service(model: BaseModel, **kw) -> PurposeRiskService:
    return PurposeRiskService(QwenController(model, max_attempts=2, in_flight=kw.pop("in_flight", 2)))


def the_run(env):
    [run] = sorted((env["tmp"] / "runs").iterdir())
    return run


def rows_by_id(run) -> dict[int, dict]:
    rows = loader.read_rows(run)
    by_id = {r["dsld_id"]: r for r in rows}
    assert len(by_id) == len(rows), "an id was written twice"
    return by_id


# --- one row per label ------------------------------------------------------


def test_a_run_writes_one_row_per_label_with_the_status_it_earned(env):
    labels = label_set(["good one", "bad two", "boom three", "long four", "good five"])

    with service(ByPassage()) as svc:
        svc.run(labels)

    rows = rows_by_id(the_run(env))
    assert {i: r["status"] for i, r in rows.items()} == {
        1: "ok", 2: "failed_checks", 3: "error", 4: "too_long", 5: "ok",
    }


def test_ok_rows_carry_the_parsed_terms_and_failed_rows_the_raw_text(env):
    labels = label_set(["good one", "bad two"])

    with service(ByPassage()) as svc:
        svc.run(labels)

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
    labels = label_set(["boom one", "good two"])
    model = ByPassage()

    with service(model, in_flight=1) as svc:
        svc.run(labels)

    rows = rows_by_id(the_run(env))
    assert (rows[1]["status"], rows[1]["attempts"], rows[1]["text"]) == ("error", 2, "")
    assert rows[2]["status"] == "ok"
    assert model.requests == 3


def test_every_row_and_the_manifest_carry_one_config_hash(env):
    labels = label_set(["good one", "good two"])

    with service(ByPassage()) as svc:
        svc.run(labels, limit=2, order="sorted")

    run = the_run(env)
    manifest = loader.read_manifest(run)
    assert manifest["label_set"] == labels
    assert manifest["passages_sha"] == loader.load_flattened(labels)[0]
    assert manifest["selection"] == {"ids": None, "limit": 2, "order": "sorted", "seed": None}
    assert {r["config_hash"] for r in loader.read_rows(run)} == {manifest["config_hash"]}


def test_the_run_finishes_with_one_notice_and_one_printed_line(env, capsys):
    labels = label_set(["good one", "bad two"])

    with service(ByPassage()) as svc:
        svc.run(labels)

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
    labels = label_set([f"good {i}" for i in range(1, 9)])

    with pytest.raises(KeyboardInterrupt):
        with service(ByPassage(kill_after=3), in_flight=1) as svc:
            svc.run(labels, limit=6)
    run = the_run(env)
    assert set(rows_by_id(run)) == {1, 2, 3}

    with service(ByPassage()) as svc:
        svc.run(run=run)

    assert set(rows_by_id(run)) == {1, 2, 3, 4, 5, 6}  # the manifest's limit, not 6 more


def test_a_resume_under_a_changed_prompt_is_refused(env, monkeypatch):
    labels = label_set(["good one", "good two"])
    with service(ByPassage()) as svc:
        svc.run(labels, limit=1)
    run = the_run(env)

    real = prompts.load
    monkeypatch.setattr(prompts, "load", lambda pid: replace(real(pid), system="a new prompt"))

    with pytest.raises(ValueError, match="config_hash"):
        with service(ByPassage()) as svc:
            svc.run(run=run)
    assert set(rows_by_id(run)) == {1}


def test_a_resume_takes_its_ids_from_the_run_not_the_call(env):
    labels = label_set(["good one"])
    with service(ByPassage()) as svc:
        svc.run(labels)

    with pytest.raises(ValueError, match="come from the run"):
        with service(ByPassage()) as svc:
            svc.run(run=the_run(env), limit=5)


# --- choosing the run's ids -------------------------------------------------


def test_a_new_run_stores_its_ids_in_run_order(env):
    labels = label_set([f"good {i}" for i in range(1, 6)])

    with service(ByPassage()) as svc:
        svc.run(labels, ids=[5, 2, 4])

    assert loader.read_ids(the_run(env)) == [5, 2, 4]
    assert set(rows_by_id(the_run(env))) == {2, 4, 5}


def test_a_random_order_is_redrawn_by_its_seed(env):
    labels = label_set([f"good {i}" for i in range(1, 9)])

    with service(ByPassage()) as svc:
        svc.run(labels, order="random", seed=7, limit=4)
        svc.run(labels, order="random", seed=7, limit=4)

    a, b = sorted((env["tmp"] / "runs").iterdir())
    assert loader.read_ids(a) == loader.read_ids(b)
    assert len(loader.read_ids(a)) == 4


def test_a_random_order_without_a_seed_is_refused(env):
    labels = label_set(["good one", "good two"])

    with pytest.raises(ValueError, match="seed"):
        with service(ByPassage()) as svc:
            svc.run(labels, order="random")
    assert not (env["tmp"] / "runs").exists()


def test_ids_outside_the_set_are_refused_before_a_run_exists(env):
    labels = label_set(["good one", "good two"])

    with pytest.raises(ValueError, match="not in label set"):
        with service(ByPassage()) as svc:
            svc.run(labels, ids=[1, 99])
    assert not (env["tmp"] / "runs").exists()


def test_a_run_is_named_after_its_label_set(env):
    labels = label_set(["good one"])

    with service(ByPassage()) as svc:
        svc.run(labels)

    assert the_run(env).name.startswith("test-set-")
    assert the_run(env).name.endswith("-0")


def test_a_resume_on_changed_passages_is_refused(env):
    labels = label_set(["good one", "good two"])
    with service(ByPassage()) as svc:
        svc.run(labels, limit=1)
    run = the_run(env)
    rows_p = store.LABELS_ROOT / labels / store.LABEL_SET_ROWS_NAME
    manifest_p = store.LABELS_ROOT / labels / store.LABEL_SET_MANIFEST_NAME
    rows_p.write_text(rows_p.read_text().replace("good two", "good 2"))
    # and its manifest updated to match, as a careless rewrite of the set would
    m = json.loads(manifest_p.read_text())
    m["passages_sha"] = hashlib.sha256(rows_p.read_bytes()).hexdigest()[:16]
    manifest_p.write_text(json.dumps(m))
    loader.reset()

    with pytest.raises(ValueError, match="passages_sha"):
        with service(ByPassage()) as svc:
            svc.run(run=run)


# --- stopping and warning ---------------------------------------------------


def test_model_unavailable_stops_the_run_with_every_finished_row_on_disk(env):
    labels = label_set(["good one", "good two", "gone three", "good four"])

    with pytest.raises(ModelUnavailable, match="exited"):
        with service(ByPassage(), in_flight=1) as svc:
            svc.run(labels)

    assert set(rows_by_id(the_run(env))) == {1, 2}
    [(subject, body)] = env["sent"]
    assert "unavailable" in subject and "exited" in body


def test_crossing_the_failure_threshold_notifies_exactly_once(env, monkeypatch):
    engine_toml = env["tmp"] / "engine.toml"
    engine_toml.write_text("[notify]\nfail_rate = 0.2\nfail_min_labels = 4\n")
    monkeypatch.setattr(config, "ENGINE_CONFIG_PATH", engine_toml)
    labels = label_set(["bad"] * 6 + ["good"] * 2)

    with service(ByPassage(), in_flight=1) as svc:
        svc.run(labels)

    subjects = [s for s, _ in env["sent"]]
    assert sum("fail" in s for s in subjects) == 1
    assert len(subjects) == 2  # the warning, then the finish
    assert len(loader.read_rows(the_run(env))) == 8  # warned, did not stop


def test_below_the_minimum_no_warning_is_sent(env, monkeypatch):
    engine_toml = env["tmp"] / "engine.toml"
    engine_toml.write_text("[notify]\nfail_rate = 0.2\nfail_min_labels = 10\n")
    monkeypatch.setattr(config, "ENGINE_CONFIG_PATH", engine_toml)
    labels = label_set(["bad"] * 5)

    with service(ByPassage()) as svc:
        svc.run(labels)

    assert [s for s, _ in env["sent"] if "fail" in s] == []
