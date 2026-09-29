"""QwenModel: one shared model, hosted on llama-server, guarding the GPU.

Pinned here: `QwenModel()` is one shared model, and a settings change is
refused mid-run and restarts the server when idle; settings come from
`configs/qwen.toml` with explicit arguments winning and a broken file falling
back; `ready()` refuses a card without room, a load that did not reach the
GPU, and a model that answers "Hello" in garbage -- stopping the server each
time; `fits()` checks one prompt against a slot with one `/tokenize` call;
`cache_shared_prefix()` primes the slots from a sample; `submit()` keeps
`n_parallel` prompts in flight and never more.

Only public functions are called. No GPU and no server: underneath the model,
`nvidia-smi` (`subprocess.run`), the server process (`subprocess.Popen`) and
its HTTP API (`urllib.request.urlopen`) are replaced.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import threading
import time
import urllib.error
import urllib.parse

import pytest

from engine import config
from engine.models import qwen_model as qm
from engine.models.base_model import ModelUnavailable, Prompt
from engine.models.qwen_model import QwenModel


@pytest.fixture(autouse=True)
def no_shared_model_between_tests():
    """Each test starts with no QwenModel built, and leaves none behind."""
    QwenModel().close()
    yield
    QwenModel().close()


# --- settings ---------------------------------------------------------------


def test_the_committed_file_supplies_the_model_and_sampling_settings():
    model = QwenModel()

    assert model.n_parallel == 4
    assert model.slot_ctx == 5120
    assert model.min_vram_mib == 21_500
    assert (model.ubatch, model.speculative, model.draft_max) == (2048, "draft-mtp", 3)
    assert model.temperature == 0.0
    assert model.enable_thinking is False
    assert model.model_path
    assert model.server_path.endswith("llama-server")


def test_the_controller_table_is_not_a_model_setting():
    """`max_attempts` is the controller's budget; the model must not receive it."""
    assert not hasattr(QwenModel(), "max_attempts")


def test_a_missing_file_falls_back_to_the_hardcoded_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "QWEN_CONFIG_PATH", tmp_path / "absent.toml")
    model = QwenModel()

    assert model.n_parallel == 4
    assert model.model_path == ""


def test_a_malformed_file_falls_back_rather_than_raising(monkeypatch, tmp_path):
    broken = tmp_path / "qwen.toml"
    broken.write_text("[model\nn_parallel = = 8\n")
    monkeypatch.setattr(config, "QWEN_CONFIG_PATH", broken)

    assert QwenModel().n_parallel == 4


def test_a_partial_file_overrides_only_what_it_names(monkeypatch, tmp_path):
    partial = tmp_path / "qwen.toml"
    partial.write_text("[sampling]\ntemperature = 0.7\n")
    monkeypatch.setattr(config, "QWEN_CONFIG_PATH", partial)
    model = QwenModel()

    assert model.temperature == 0.7
    assert model.slot_ctx == 5120


def test_an_explicit_argument_beats_the_file(monkeypatch, tmp_path):
    file = tmp_path / "qwen.toml"
    file.write_text("[model]\nn_parallel = 6\n")
    monkeypatch.setattr(config, "QWEN_CONFIG_PATH", file)

    assert QwenModel(n_parallel=2).n_parallel == 2


def test_an_explicit_falsy_argument_still_wins(monkeypatch, tmp_path):
    """`None` is the sentinel, not falsiness: 0 and False are real settings."""
    file = tmp_path / "qwen.toml"
    file.write_text("[sampling]\nseed = 7\n")
    monkeypatch.setattr(config, "QWEN_CONFIG_PATH", file)

    assert QwenModel(seed=0).seed == 0


# --- a stand-in machine: nvidia-smi, the server process, its HTTP API --------


class FakeProc:
    def __init__(self, args):
        self.args = args
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class FakeReply:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class FakeMachine:
    """Answers nvidia-smi, starts fake servers, and answers their HTTP calls.

    Tokens are whitespace words, so a test controls prompt size by its text.
    The card is nearly full while a server runs and free otherwise, like the
    real one. `in_flight_peak` records how many completions ran at once.
    """

    def __init__(self):
        self.hello = "Hello! How can I help?"
        self.delay_s = 0.0
        self.bodies: list[dict] = []
        self.spawned: list[FakeProc] = []
        self.holders = ""  # nvidia-smi's compute-apps CSV
        self.free_mib = lambda: 5_000 if self.running() else 24_000
        self.health_failures = 0  # /health answers 503 this many times first
        self.server_log = ""  # written to the log by a server that exits at once
        self._lock = threading.Lock()
        self._in_flight = 0
        self.in_flight_peak = 0

    def running(self):
        return any(p.poll() is None for p in self.spawned)

    # subprocess.run: only nvidia-smi is ever run.
    def run(self, args, **kwargs):
        if "--query-gpu=memory.free" in args:
            return subprocess.CompletedProcess(args, 0, stdout=f"{self.free_mib()}\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout=self.holders, stderr="")

    # subprocess.Popen: the server.
    def popen(self, args, stdout=None, **kwargs):
        proc = FakeProc(args)
        if self.server_log:
            stdout.write(self.server_log)
            proc.returncode = 1
        self.spawned.append(proc)
        return proc

    # urllib.request.urlopen: the server's HTTP API.
    def urlopen(self, req, timeout=None):
        path = urllib.parse.urlparse(req.full_url).path
        body = json.loads(req.data) if req.data else None
        if path == "/health":
            if self.health_failures:
                self.health_failures -= 1
                raise urllib.error.HTTPError(req.full_url, 503, "Loading", None, io.BytesIO(b"Loading model"))
            return FakeReply({"status": "ok"})
        if path == "/tokenize":
            return FakeReply({"tokens": body["content"].split()})
        assert path == "/completion"
        with self._lock:
            self.bodies.append(body)
            self._in_flight += 1
            self.in_flight_peak = max(self.in_flight_peak, self._in_flight)
        time.sleep(self.delay_s)
        with self._lock:
            self._in_flight -= 1
        is_hello = "<|im_start|>user\nHello<|im_end|>" in body["prompt"]
        return FakeReply(
            {
                "content": self.hello if is_hello else "P: supports sleep | 4",
                "tokens_evaluated": 100,
                "tokens_predicted": 7,
                "stop_type": "eos",
            }
        )


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A FakeMachine, and a QwenModel built with real-looking paths on it."""
    fake = FakeMachine()
    monkeypatch.setattr(qm.subprocess, "run", fake.run)
    monkeypatch.setattr(qm.subprocess, "Popen", fake.popen)
    monkeypatch.setattr(qm.urllib.request, "urlopen", fake.urlopen)

    weights = tmp_path / "weights.gguf"
    weights.write_bytes(b"\0" * 1024)
    binary = tmp_path / "llama-server"
    binary.write_text("")
    fake.model = QwenModel(model_path=str(weights), server_path=str(binary), n_parallel=4, slot_ctx=5120)
    return fake


def arg(proc: FakeProc, flag: str) -> str:
    return proc.args[proc.args.index(flag) + 1]


# --- one shared model -------------------------------------------------------


def test_every_call_gets_the_same_model():
    assert QwenModel() is QwenModel()


def test_no_settings_or_the_same_settings_keep_the_model_as_it_is(machine):
    machine.model.ready()

    assert QwenModel() is machine.model
    assert QwenModel(n_parallel=4) is machine.model
    assert len(machine.spawned) == 1 and not machine.spawned[0].terminated


def test_new_settings_before_any_start_are_simply_taken(machine, caplog):
    with caplog.at_level(logging.WARNING):
        model = QwenModel(n_parallel=2)

    assert model is machine.model
    assert model.n_parallel == 2
    assert caplog.records == []


def test_new_settings_while_idle_restart_the_server_with_a_warning(machine, caplog):
    machine.model.ready()

    with caplog.at_level(logging.WARNING):
        model = QwenModel(n_parallel=2)
    model.ready()

    assert model is machine.model
    assert "shutting down" in caplog.text
    assert machine.spawned[0].terminated
    assert arg(machine.spawned[1], "-np") == "2"


def test_new_settings_mid_run_are_refused(machine):
    machine.model.ready()
    machine.delay_s = 0.2
    futures = [machine.model.submit(Prompt(system="s", user=f"label {i}")) for i in range(2)]

    with pytest.raises(RuntimeError, match="mid-run"):
        QwenModel(n_parallel=2)

    assert [f.result().text for f in futures] == ["P: supports sleep | 4"] * 2
    assert machine.model.n_parallel == 4
    assert not machine.spawned[0].terminated


def test_after_close_the_next_call_builds_a_fresh_model(machine):
    machine.model.close()

    fresh = QwenModel()

    assert fresh is not machine.model
    assert fresh.model_path != machine.model.model_path  # from the file again


def test_a_closed_model_cannot_start_once_another_is_built(machine):
    machine.model.close()
    QwenModel()

    with pytest.raises(ModelUnavailable, match="closed"):
        machine.model.ready()


def test_a_closed_model_can_start_again_if_nothing_replaced_it(machine):
    machine.model.ready()
    machine.model.close()

    machine.model.ready()

    assert QwenModel() is machine.model
    assert len(machine.spawned) == 2


# --- the server's command line ----------------------------------------------


def test_the_server_gets_one_slot_ctx_per_slot(machine):
    machine.model.ready()
    [proc] = machine.spawned

    assert arg(proc, "-np") == "4"
    assert arg(proc, "-c") == str(4 * 5120)
    assert "--no-kv-unified" in proc.args


def test_the_server_gets_the_tuned_batch_and_speculation(machine):
    QwenModel(ubatch=2048, speculative="draft-mtp", draft_max=3).ready()
    [proc] = machine.spawned

    assert arg(proc, "-ub") == "2048"
    assert arg(proc, "--spec-type") == "draft-mtp"
    assert arg(proc, "--spec-draft-n-max") == "3"


def test_speculation_can_be_turned_off(machine):
    QwenModel(speculative="none").ready()

    assert "--spec-type" not in machine.spawned[0].args


# --- ready() ----------------------------------------------------------------


def test_ready_starts_the_server_and_says_hello_once(machine):
    machine.model.ready()
    machine.model.ready()

    assert len(machine.spawned) == 1
    assert len(machine.bodies) == 1  # the hello


@pytest.mark.parametrize(
    "hello",
    ["Hello! How can I help you today?", "Hi there.", "  Hello — nice to meet you.  "],
)
def test_a_greeting_passes_the_hello_check(machine, hello):
    machine.hello = hello

    machine.model.ready()


@pytest.mark.parametrize(
    "hello, why",
    [
        ("", "empty"),
        ("   \n  ", "empty"),
        ("@@@@ #### !!!! %%%% ^^^^", "no words"),
        ("ĠĠĠ ĊĊĊ ▁▁▁ ĉĉ", "no words"),
        ("hello " * 1000, "too long"),
    ],
)
def test_ready_refuses_a_model_that_answers_in_garbage(machine, hello, why):
    machine.hello = hello

    with pytest.raises(ModelUnavailable, match="not coherent") as caught:
        machine.model.ready()

    assert why in str(caught.value)
    assert machine.spawned[0].terminated, "a refused server must not keep the card"


def test_a_refused_model_starts_fresh_next_time(machine):
    machine.hello = ""
    with pytest.raises(ModelUnavailable):
        machine.model.ready()

    machine.hello = "Hello!"
    machine.model.ready()

    assert len(machine.spawned) == 2


def test_ready_refuses_without_a_model_path(machine):
    QwenModel(model_path="")

    with pytest.raises(ModelUnavailable, match="model_path"):
        machine.model.ready()
    assert machine.spawned == []


def test_ready_refuses_without_a_server_binary(machine, tmp_path):
    QwenModel(server_path=str(tmp_path / "nowhere"))

    with pytest.raises(ModelUnavailable, match="llama-server"):
        machine.model.ready()


def test_ready_refuses_a_full_card_and_names_the_holder(machine):
    machine.free_mib = lambda: 6_690
    machine.holders = "12345, python, 17214\n"

    with pytest.raises(ModelUnavailable) as caught:
        machine.model.ready()

    assert "6690 MiB" in str(caught.value)
    assert "kill 12345" in str(caught.value)
    assert machine.spawned == []


def test_ready_refuses_a_load_that_did_not_reach_the_gpu(machine, tmp_path):
    """A CPU fallback loads fine and takes almost no VRAM -- only the clock tells."""
    big = tmp_path / "big.gguf"
    with big.open("wb") as f:
        f.truncate(4 * 2**30)  # 4 GiB, sparse
    QwenModel(model_path=str(big))
    frees = iter([24_000, 23_700])
    machine.free_mib = lambda: next(frees)

    with pytest.raises(ModelUnavailable, match="not on the GPU"):
        machine.model.ready()
    assert machine.spawned[0].terminated


def test_a_server_that_dies_while_loading_is_reported_with_its_log(machine):
    machine.server_log = "CUDA error: out of memory\n"

    with pytest.raises(ModelUnavailable) as caught:
        machine.model.ready()

    assert "exited with code 1" in str(caught.value)
    assert "out of memory" in str(caught.value)


def test_ready_waits_out_the_servers_loading_answer(machine, monkeypatch):
    """The real server answers /health with 503 "Loading model" for ~10s."""
    machine.health_failures = 2
    monkeypatch.setattr(qm.time, "sleep", lambda s: None)

    machine.model.ready()

    assert len(machine.spawned) == 1


def test_close_stops_the_server(machine):
    machine.model.ready()
    machine.model.close()
    machine.model.close()  # idempotent

    assert machine.spawned[0].terminated


# --- generation -------------------------------------------------------------


def test_generate_sends_the_rendered_prompt_and_sampling_settings(machine):
    g = machine.model.generate(Prompt(system="sys", user="label text", stop=("\nEND",), max_tokens=400))

    body = machine.bodies[-1]
    assert body["prompt"].startswith("<|im_start|>system\nsys<|im_end|>")
    assert body["prompt"].endswith("<think>\n\n</think>\n\n")
    assert body["n_predict"] == 400
    assert body["stop"] == ["\nEND"]
    assert body["temperature"] == 0.0
    assert body["cache_prompt"] is True
    assert "json_schema" not in body
    assert g.text == "P: supports sleep | 4"
    assert (g.usage.prompt_tokens, g.usage.completion_tokens) == (100, 7)
    assert g.stop_reason == "eos"


def test_a_schema_is_sent_as_the_servers_json_schema(machine):
    schema = {"type": "object"}

    machine.model.generate(Prompt(system="s", user="u", schema=schema))

    assert machine.bodies[-1]["json_schema"] == schema


def test_submit_keeps_every_slot_busy_and_no_more(machine):
    machine.model.ready()
    machine.delay_s = 0.1

    started = time.monotonic()
    futures = [machine.model.submit(Prompt(system="s", user=f"label {i}")) for i in range(8)]
    out = [f.result() for f in futures]

    assert len(out) == 8
    assert machine.in_flight_peak == 4  # n_parallel, never more
    assert time.monotonic() - started < 0.35  # two rounds of 0.1s, not eight


def test_close_cancels_what_is_still_queued(machine):
    machine.model.ready()
    machine.delay_s = 0.2
    futures = [machine.model.submit(Prompt(system="s", user=f"label {i}")) for i in range(8)]

    machine.model.close()

    assert any(f.cancelled() for f in futures)


# --- fitting ----------------------------------------------------------------


def prompt_of(n_words: int, max_tokens: int = 400) -> Prompt:
    return Prompt(system="s", user=" ".join(["w"] * n_words), max_tokens=max_tokens)


def test_fits_checks_one_prompt_against_a_slot(machine):
    """Prompt + output budget + repair margin must fit, or it is never sent."""
    assert machine.model.fits(prompt_of(100, max_tokens=400))
    assert not machine.model.fits(prompt_of(4800, max_tokens=400))  # 4800 + 400 + 64 > 5120


def test_the_output_budget_counts_toward_the_fit(machine):
    assert machine.model.fits(prompt_of(3100, max_tokens=400))
    assert not machine.model.fits(prompt_of(3100, max_tokens=2000))  # 3100 + 2000 + 64 > 5120


def test_fits_asks_the_server_once_and_generates_nothing(machine, monkeypatch):
    machine.model.ready()
    calls = []
    real = machine.urlopen
    monkeypatch.setattr(
        qm.urllib.request,
        "urlopen",
        lambda req, timeout=None: calls.append(urllib.parse.urlparse(req.full_url).path)
        or real(req, timeout),
    )

    machine.model.fits(prompt_of(10))

    assert calls == ["/tokenize"]


# --- caching the shared prefix ----------------------------------------------


def primes_sent_by(machine, sample) -> list[dict]:
    """The completion bodies one `cache_shared_prefix()` sends to warm the slots."""
    before = len(machine.bodies)
    machine.model.cache_shared_prefix(sample)
    return machine.bodies[before:]


def test_every_slot_caches_the_shared_system_turn(machine):
    """Cut after the last full turn, so it tokenises as inside each prompt."""
    machine.model.ready()
    primes = primes_sent_by(machine, [Prompt(system="sys", user=f"label {i}") for i in range(3)])

    assert len(primes) == 4  # one per slot
    assert all(b["prompt"] == "<|im_start|>system\nsys<|im_end|>\n" for b in primes)
    assert all(b["n_predict"] == 1 for b in primes)


def test_nothing_is_cached_for_a_single_prompt_or_different_systems(machine):
    machine.model.ready()

    assert primes_sent_by(machine, [Prompt(system="sys", user="only")]) == []
    assert primes_sent_by(machine, [Prompt(system="A", user="x"), Prompt(system="B", user="x")]) == []


def test_a_prefix_the_slots_already_hold_is_not_sent_again(machine):
    sample = [Prompt(system="sys", user=f"label {i}") for i in range(3)]
    machine.model.cache_shared_prefix(sample)

    assert primes_sent_by(machine, sample) == []


def test_a_restarted_server_caches_the_prefix_afresh(machine):
    sample = [Prompt(system="sys", user=f"label {i}") for i in range(3)]
    machine.model.cache_shared_prefix(sample)
    machine.model.close()
    machine.model.ready()

    assert len(primes_sent_by(machine, sample)) == 4
