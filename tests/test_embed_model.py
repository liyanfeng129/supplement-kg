"""EmbedModel: one shared embedding model, hosted on llama-server, guarding the GPU.

Pinned here: `EmbedModel()` is one shared model; settings come from
`configs/embed.toml` with explicit arguments winning and a broken file falling
back; each of `n_servers` servers gets last-token pooling, L2 normalisation and
one slot; `ready()` settles the end token once and refuses a card without room
for every server, a load that did not reach the GPU, malformed vectors and
similarity that makes no sense -- stopping every server each time; a request
goes to the least-busy server, and one server exiting stops the whole model; `embed()` applies the instruction
template and returns rows in input order; `fits()` makes no call for a text
plainly short enough; a failed request is the request's own error while the
server is healthy, and `ModelUnavailable` once it is not.

Only public functions are called. No GPU and no server: underneath the model,
`nvidia-smi` (`subprocess.run`), the server process (`subprocess.Popen`) and
its HTTP API (`urllib.request.urlopen`) are replaced.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import threading
import time
import urllib.error
import urllib.parse

import numpy as np
import pytest

from engine import config
from engine.models import embed_model as em
from engine.models.base_model import ModelUnavailable
from engine.models.embed_model import EmbedModel

EOS = "<|endoftext|>"
EOS_ID = 151643


@pytest.fixture(autouse=True)
def no_shared_model_between_tests():
    """Each test starts with no EmbedModel built, and leaves none behind."""
    EmbedModel().close()
    yield
    EmbedModel().close()


# --- settings ---------------------------------------------------------------


def test_the_committed_file_supplies_the_settings():
    model = EmbedModel()

    assert (model.n_servers, model.n_ctx, model.ubatch) == (3, 4096, 4096)
    assert model.min_vram_mib == 5200
    assert model.model_path.endswith("Qwen3-Embedding-0.6B-f16.gguf")
    assert model.server_path.endswith("llama-server")


def test_the_controller_table_is_not_a_model_setting():
    model = EmbedModel()

    assert not hasattr(model, "batch")
    assert not hasattr(model, "in_flight")


def test_a_missing_file_falls_back_to_the_hardcoded_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EMBED_CONFIG_PATH", tmp_path / "absent.toml")
    model = EmbedModel()

    assert model.n_servers == 1
    assert model.model_path == ""


def test_a_malformed_file_falls_back_rather_than_raising(monkeypatch, tmp_path):
    broken = tmp_path / "embed.toml"
    broken.write_text("[model\nn_servers = = 8\n")
    monkeypatch.setattr(config, "EMBED_CONFIG_PATH", broken)

    assert EmbedModel().n_servers == 1


def test_an_explicit_argument_beats_the_file():
    assert EmbedModel(n_servers=2).n_servers == 2


def test_slots_are_no_longer_a_setting():
    assert not hasattr(EmbedModel(), "n_parallel")


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


def bag_of_words(text: str, dim: int = 1024) -> list[float]:
    """A unit vector from the text's words: shared words, similar vectors."""
    vector = np.zeros(dim)
    for word in text.lower().split():
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    return list(vector / np.linalg.norm(vector))


class FakeMachine:
    """Answers nvidia-smi, starts fake servers, and answers their HTTP calls.

    Tokens are whitespace words, and `<|endoftext|>` is one token wherever it
    appears. `adds_eos` is whether the tokenizer appends it itself, as a GGUF
    with `add_eos_token` does. `vectors` turns each input into its embedding.
    """

    def __init__(self):
        self.adds_eos = False
        self.vectors = bag_of_words
        self.reverse = False  # answer `data` in reverse order
        # Each running server takes 4,000 MiB of a 24,000 MiB card.
        self.free_mib = lambda: 24_000 - 4_000 * self.running()
        self.spawned: list[FakeProc] = []
        self.tokenize_calls: list[dict] = []
        self.embed_bodies: list[dict] = []
        self.embed_ports: list[int] = []  # which server answered each embedding request
        self.embed_error = None
        self.health_down = False
        self.delay_s = 0.0
        self._lock = threading.Lock()
        self._in_flight = 0
        self.in_flight_peak = 0

    def running(self) -> int:
        return sum(p.poll() is None for p in self.spawned)

    def run(self, args, **kwargs):
        if "--query-gpu=memory.free" in args:
            return subprocess.CompletedProcess(args, 0, stdout=f"{self.free_mib()}\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def popen(self, args, stdout=None, **kwargs):
        proc = FakeProc(args)
        self.spawned.append(proc)
        return proc

    def tokens(self, content: str, add_special: bool) -> list[int]:
        words = content.replace(EOS, f" {EOS} ").split()
        ids = [EOS_ID if w == EOS else 1000 + len(w) for w in words]
        return ids + [EOS_ID] if add_special and self.adds_eos else ids

    def urlopen(self, req, timeout=None):
        path = urllib.parse.urlparse(req.full_url).path
        body = json.loads(req.data) if req.data else None
        if path == "/health":
            if self.health_down:
                raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
            return FakeReply({"status": "ok"})
        if path == "/tokenize":
            self.tokenize_calls.append(body)
            return FakeReply({"tokens": self.tokens(body["content"], body.get("add_special", False))})
        assert path == "/v1/embeddings"
        with self._lock:
            self.embed_bodies.append(body)
            self.embed_ports.append(urllib.parse.urlparse(req.full_url).port)
            self._in_flight += 1
            self.in_flight_peak = max(self.in_flight_peak, self._in_flight)
        time.sleep(self.delay_s)
        with self._lock:
            self._in_flight -= 1
        if self.embed_error is not None:
            raise self.embed_error
        data = [
            {"index": i, "embedding": self.vectors(text)} for i, text in enumerate(body["input"])
        ]
        tokens = sum(len(self.tokens(t, True)) for t in body["input"])
        return FakeReply(
            {
                "data": data[::-1] if self.reverse else data,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A FakeMachine, and an EmbedModel built with real-looking paths on it."""
    fake = FakeMachine()
    monkeypatch.setattr(em.subprocess, "run", fake.run)
    monkeypatch.setattr(em.subprocess, "Popen", fake.popen)
    monkeypatch.setattr(em.urllib.request, "urlopen", fake.urlopen)

    weights = tmp_path / "weights.gguf"
    weights.write_bytes(b"\0" * 1024)
    binary = tmp_path / "llama-server"
    binary.write_text("")
    fake.model = EmbedModel(model_path=str(weights), server_path=str(binary), n_servers=1, ubatch=64)
    return fake


def arg(proc: FakeProc, flag: str) -> str:
    return proc.args[proc.args.index(flag) + 1]


# --- one shared model -------------------------------------------------------


def test_every_call_gets_the_same_model():
    assert EmbedModel() is EmbedModel()


def test_new_settings_while_idle_restart_the_server(machine):
    machine.model.ready()

    EmbedModel(n_servers=2).ready()

    assert machine.spawned[0].terminated
    assert len(machine.spawned) == 3


def test_new_settings_mid_run_are_refused(machine):
    machine.model.ready()
    machine.delay_s = 0.2
    future = machine.model.submit(["a label"])

    with pytest.raises(RuntimeError, match="mid-run"):
        EmbedModel(n_servers=2)

    assert future.result().vectors.shape == (1, 1024)
    assert not machine.spawned[0].terminated


# --- the server's command line ----------------------------------------------


def test_each_server_pools_the_last_token_normalises_and_runs_one_slot(machine):
    model = EmbedModel(n_servers=3, n_ctx=8192, ubatch=8192)
    model.ready()

    assert len(machine.spawned) == 3
    for proc in machine.spawned:
        assert "--embedding" in proc.args
        assert arg(proc, "--pooling") == "last"
        assert arg(proc, "--embd-normalize") == "2"
        assert "--kv-unified" not in proc.args
        assert (arg(proc, "-np"), arg(proc, "-c")) == ("1", "8192")
        assert (arg(proc, "-ub"), arg(proc, "-b")) == ("8192", "8192")
    assert len({arg(proc, "--port") for proc in machine.spawned}) == 3
    assert len(model.log_paths) == 3


# --- ready() ----------------------------------------------------------------


def test_ready_refuses_a_card_without_room(machine):
    machine.free_mib = lambda: 2_000

    with pytest.raises(ModelUnavailable, match="2000 MiB of VRAM free"):
        machine.model.ready()

    assert machine.spawned == []


def test_ready_refuses_a_card_without_room_for_every_server(machine):
    # 24,000 MiB free: room for four servers of 5,200, not five.
    with pytest.raises(ModelUnavailable, match="5 servers"):
        EmbedModel(n_servers=5).ready()

    assert machine.spawned == []


def test_every_server_must_take_its_weights_onto_the_gpu(machine):
    # The third load takes nothing: two servers' worth is all the card ever gives up.
    machine.free_mib = lambda: 24_000 - 4_000 * min(machine.running(), 2)

    with pytest.raises(ModelUnavailable, match="not on the GPU"):
        EmbedModel(n_servers=3, min_vram_mib=1).ready()

    assert len(machine.spawned) == 3
    assert all(proc.terminated for proc in machine.spawned)


def test_the_end_token_is_settled_once_for_all_servers(machine):
    EmbedModel(n_servers=3).ready()

    probes = [call for call in machine.tokenize_calls if call["content"] == "probe"]
    assert len(probes) == 1


def test_ready_refuses_a_load_that_did_not_reach_the_gpu(machine):
    machine.free_mib = lambda: 24_000  # the load took nothing

    with pytest.raises(ModelUnavailable, match="not on the GPU"):
        EmbedModel(min_vram_mib=1).ready()

    assert machine.spawned[0].terminated


def test_ready_refuses_vectors_of_the_wrong_size(machine):
    machine.vectors = lambda text: bag_of_words(text, dim=768)

    with pytest.raises(ModelUnavailable, match="768"):
        machine.model.ready()

    assert machine.spawned[0].terminated


def test_ready_refuses_vectors_that_are_not_unit_length(machine):
    machine.vectors = lambda text: [2 * x for x in bag_of_words(text)]

    with pytest.raises(ModelUnavailable, match="unit length"):
        machine.model.ready()


def test_ready_refuses_similarity_that_makes_no_sense(machine):
    """Broken pooling: every text gets the same well-formed vector."""
    machine.vectors = lambda text: bag_of_words("the same for everything")

    with pytest.raises(ModelUnavailable, match="similar"):
        machine.model.ready()

    assert machine.spawned[0].terminated


def test_ready_refuses_a_server_that_has_since_exited(machine):
    machine.model.ready()
    machine.spawned[0].returncode = 1

    with pytest.raises(ModelUnavailable, match="exited"):
        machine.model.ready()

    assert len(machine.spawned) == 1


# --- the end token and the template -----------------------------------------


def test_the_end_token_is_appended_when_the_tokenizer_does_not(machine):
    machine.model.embed(["vitamin c"])

    assert machine.embed_bodies[-1]["input"] == [f"vitamin c{EOS}"]


def test_the_end_token_is_not_doubled_when_the_tokenizer_adds_it(machine):
    machine.adds_eos = True

    machine.model.embed(["vitamin c"])

    assert machine.embed_bodies[-1]["input"] == ["vitamin c"]


def test_an_instruction_is_laid_out_in_the_model_s_template(machine):
    machine.adds_eos = True

    machine.model.embed(["vitamin c"], instruction="Find the product")

    assert machine.embed_bodies[-1]["input"] == ["Instruct: Find the product\nQuery:vitamin c"]


# --- embed() ----------------------------------------------------------------


def test_embed_returns_rows_in_input_order_with_the_server_s_token_count(machine):
    machine.model.ready()
    machine.reverse = True

    embedding = machine.model.embed(["zinc", "iron tablets"])

    assert embedding.vectors.dtype == np.float32
    assert embedding.vectors.shape == (2, 1024)
    assert np.allclose(embedding.vectors[0], bag_of_words(f"zinc{EOS}"))
    assert np.allclose(embedding.vectors[1], bag_of_words(f"iron tablets{EOS}"))
    assert embedding.usage.prompt_tokens == 5  # zinc, eos; iron, tablets, eos
    assert embedding.usage.completion_tokens == 0


def test_a_failed_request_on_a_healthy_server_raises_its_own_error(machine):
    machine.model.ready()
    machine.embed_error = urllib.error.HTTPError("u", 400, "Bad", None, io.BytesIO(b"bad input"))

    with pytest.raises(em.ServerError, match="bad input"):
        machine.model.embed(["zinc"])


def test_a_failed_request_on_a_dead_server_is_model_unavailable(machine):
    machine.model.ready()
    machine.embed_error = urllib.error.URLError("refused")
    machine.health_down = True

    with pytest.raises(ModelUnavailable, match="health"):
        machine.model.embed(["zinc"])


# --- fits() -----------------------------------------------------------------


def test_fits_makes_no_call_for_a_text_plainly_short_enough(machine):
    machine.model.ready()
    calls = len(machine.tokenize_calls)

    assert machine.model.fits("x" * 40, instruction="Find")  # 40 + 15 + 1 <= 64
    assert machine.model.fits("x" * 63)  # 63 + 1 == 64, the boundary

    assert len(machine.tokenize_calls) == calls
    machine.model.fits("x" * 64)
    assert len(machine.tokenize_calls) == calls + 1


def test_fits_counts_tokens_with_the_end_token_for_a_long_text(machine):
    machine.model.ready()

    assert machine.model.fits(" ".join(["w"] * 63))  # 63 words + the end token
    assert not machine.model.fits(" ".join(["w"] * 64))
    assert machine.tokenize_calls[-1]["content"].endswith(EOS)


# --- submit() and close() ---------------------------------------------------


def test_submit_runs_at_most_one_request_per_server_at_once(machine):
    model = EmbedModel(n_servers=3)
    model.ready()
    machine.delay_s = 0.05

    futures = [model.submit([f"label {i}"]) for i in range(12)]

    assert [f.result().vectors.shape for f in futures] == [(1, 1024)] * 12
    assert machine.in_flight_peak == 3


def test_requests_in_flight_together_land_on_different_servers(machine):
    model = EmbedModel(n_servers=3)
    model.ready()
    machine.embed_ports.clear()
    machine.delay_s = 0.2

    futures = [model.submit([f"label {i}"]) for i in range(3)]
    [f.result() for f in futures]

    ports = {int(arg(proc, "--port")) for proc in machine.spawned}
    assert set(machine.embed_ports) == ports


def test_an_idle_model_sends_each_request_to_the_first_server(machine):
    model = EmbedModel(n_servers=3)
    model.ready()
    machine.embed_ports.clear()

    model.embed(["zinc"])
    model.embed(["iron"])

    assert machine.embed_ports == [int(arg(machine.spawned[0], "--port"))] * 2


def test_one_server_exiting_stops_the_whole_model(machine):
    model = EmbedModel(n_servers=3)
    model.ready()
    machine.spawned[1].returncode = 1

    with pytest.raises(ModelUnavailable, match="exited"):
        model.embed(["zinc"])

    model.close()
    assert machine.spawned[0].terminated and machine.spawned[2].terminated
    assert len(machine.spawned) == 3


def test_close_stops_every_server_and_is_idempotent(machine):
    model = EmbedModel(n_servers=3)
    model.ready()

    model.close()
    model.close()

    assert all(proc.terminated for proc in machine.spawned)
