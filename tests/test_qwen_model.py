"""QwenModel hosts itself on llama-server, guards the GPU, and runs slots in parallel.

Pinned here: settings come from the committed `configs/qwen.toml` with explicit
arguments winning and a broken file falling back; `ready()` refuses a card
without room, a load that did not reach the GPU, and a model that answers
"Hello" in garbage -- stopping the server each time so nothing is left holding
the card; `plan()` measures a work set and names what cannot fit a slot;
`submit()` keeps `n_parallel` prompts in flight and never more.

No GPU and no server: the process and the HTTP calls are replaced.
"""

from __future__ import annotations

import threading
import time

import pytest

from engine.models import qwen_model as qm
from engine.models.base_model import ModelUnavailable, Plan, Prompt
from engine.models.qwen_model import QwenModel, load_defaults

# --- settings ---------------------------------------------------------------


def test_the_committed_file_supplies_the_model_and_sampling_settings():
    defaults = load_defaults()

    assert defaults["n_parallel"] == 4
    assert defaults["slot_ctx"] == 5120
    assert defaults["min_vram_mib"] == 21_500
    assert (defaults["ubatch"], defaults["speculative"], defaults["draft_max"]) == (2048, "draft-mtp", 3)
    assert defaults["temperature"] == 0.0
    assert defaults["enable_thinking"] is False
    assert defaults["model_path"]
    assert defaults["server_path"].endswith("llama-server")


def test_the_controller_table_is_not_a_model_setting():
    """`max_attempts` is the controller's budget; the model must not receive it."""
    assert "max_attempts" not in load_defaults()


def test_a_missing_file_falls_back_to_the_hardcoded_defaults(tmp_path):
    defaults = load_defaults(tmp_path / "absent.toml")

    assert defaults["n_parallel"] == 4
    assert defaults["model_path"] == ""


def test_a_malformed_file_falls_back_rather_than_raising(tmp_path):
    broken = tmp_path / "qwen.toml"
    broken.write_text("[model\nn_parallel = = 8\n")

    assert load_defaults(broken)["n_parallel"] == 4


def test_a_partial_file_overrides_only_what_it_names(tmp_path):
    partial = tmp_path / "qwen.toml"
    partial.write_text("[sampling]\ntemperature = 0.7\n")

    defaults = load_defaults(partial)

    assert defaults["temperature"] == 0.7
    assert defaults["slot_ctx"] == 5120


def test_an_explicit_argument_beats_the_file(monkeypatch):
    monkeypatch.setattr(
        qm, "load_defaults", lambda *a, **k: {**qm._FALLBACKS, "n_parallel": 6}
    )

    assert QwenModel().n_parallel == 6
    assert QwenModel(n_parallel=2).n_parallel == 2


def test_an_explicit_falsy_argument_still_wins(monkeypatch):
    """`None` is the sentinel, not falsiness: 0 and False are real settings."""
    monkeypatch.setattr(qm, "load_defaults", lambda *a, **k: {**qm._FALLBACKS, "seed": 7})

    assert QwenModel(seed=0).seed == 0


def test_the_server_gets_one_slot_ctx_per_slot(tmp_path):
    model = QwenModel(n_parallel=4, slot_ctx=5120)

    args = model._server_args(port=9999)

    assert args[args.index("-np") + 1] == "4"
    assert args[args.index("-c") + 1] == str(4 * 5120)
    assert "--no-kv-unified" in args


def test_the_server_gets_the_tuned_batch_and_speculation():
    args = QwenModel(ubatch=2048, speculative="draft-mtp", draft_max=3)._server_args(port=9999)

    assert args[args.index("-ub") + 1] == "2048"
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    assert args[args.index("--spec-draft-n-max") + 1] == "3"


def test_speculation_can_be_turned_off():
    assert "--spec-type" not in QwenModel(speculative="none")._server_args(port=9999)


# --- the hello check --------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["Hello! How can I help you today?", "Hi there.", "  Hello — nice to meet you.  "],
)
def test_a_greeting_is_coherent(text):
    assert qm.incoherence(text) is None


@pytest.mark.parametrize(
    "text, why",
    [
        ("", "empty"),
        ("   \n  ", "empty"),
        ("@@@@ #### !!!! %%%% ^^^^", "no words"),
        ("ĠĠĠ ĊĊĊ ▁▁▁ ĉĉ", "no words"),
        ("hello " * 1000, "too long"),
    ],
)
def test_garbage_is_not_coherent(text, why):
    assert why in qm.incoherence(text)


# --- a stand-in server ------------------------------------------------------


class FakeProc:
    def __init__(self):
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


class FakeServer:
    """Answers the model's HTTP calls: health, tokenize, completion.

    Tokens are whitespace words, so a test controls prompt size by its text.
    `in_flight_peak` records how many completions ran at once.
    """

    def __init__(self, hello="Hello! How can I help?", delay_s=0.0):
        self.hello = hello
        self.delay_s = delay_s
        self.bodies: list[dict] = []
        self.spawned: list[FakeProc] = []
        self._lock = threading.Lock()
        self._in_flight = 0
        self.in_flight_peak = 0

    def request(self, path, body=None, timeout=None):
        if path == "/health":
            return {"status": "ok"}
        if path == "/tokenize":
            return {"tokens": body["content"].split()}
        assert path == "/completion"
        with self._lock:
            self.bodies.append(body)
            self._in_flight += 1
            self.in_flight_peak = max(self.in_flight_peak, self._in_flight)
        time.sleep(self.delay_s)
        with self._lock:
            self._in_flight -= 1
        is_hello = "<|im_start|>user\nHello<|im_end|>" in body["prompt"]
        return {
            "content": self.hello if is_hello else "P: supports sleep | 4",
            "tokens_evaluated": 100,
            "tokens_predicted": 7,
            "stop_type": "eos",
        }


@pytest.fixture
def server(monkeypatch, tmp_path):
    """A model wired to a FakeServer, with a free card and real-looking paths."""
    fake = FakeServer()
    weights = tmp_path / "weights.gguf"
    weights.write_bytes(b"\0" * 1024)
    binary = tmp_path / "llama-server"
    binary.write_text("")

    # Like the real card: nearly full while a server is running, free otherwise.
    running = lambda: any(p.poll() is None for p in fake.spawned)  # noqa: E731
    monkeypatch.setattr(qm, "free_vram_mib", lambda: 5_000 if running() else 24_000)
    monkeypatch.setattr(qm, "gpu_holders", lambda: [])

    def spawn(self, args, log, env):
        proc = FakeProc()
        fake.spawned.append(proc)
        return proc

    monkeypatch.setattr(QwenModel, "_spawn", spawn)
    monkeypatch.setattr(QwenModel, "_request", lambda self, *a, **k: fake.request(*a, **k))
    fake.model = QwenModel(model_path=str(weights), server_path=str(binary), n_parallel=4, slot_ctx=5120)
    return fake


# --- lifecycle --------------------------------------------------------------


def test_ready_starts_the_server_and_says_hello_once(server):
    server.model.ready()
    server.model.ready()

    assert len(server.spawned) == 1
    assert len(server.bodies) == 1  # the hello


def test_ready_refuses_a_model_that_answers_in_garbage(server):
    server.hello = "ĠĠĠ ĊĊĊ ▁▁▁ ĉĉ"

    with pytest.raises(ModelUnavailable, match="not coherent"):
        server.model.ready()

    assert server.spawned[0].terminated, "a refused server must not keep the card"


def test_a_refused_model_starts_fresh_next_time(server):
    server.hello = ""
    with pytest.raises(ModelUnavailable):
        server.model.ready()

    server.hello = "Hello!"
    server.model.ready()

    assert len(server.spawned) == 2


def test_ready_refuses_without_a_model_path(server):
    server.model.model_path = ""

    with pytest.raises(ModelUnavailable, match="model_path"):
        server.model.ready()
    assert server.spawned == []


def test_ready_refuses_without_a_server_binary(server, tmp_path):
    server.model.server_path = str(tmp_path / "nowhere")

    with pytest.raises(ModelUnavailable, match="llama-server"):
        server.model.ready()


def test_ready_refuses_a_full_card_and_names_the_holder(server, monkeypatch):
    monkeypatch.setattr(qm, "free_vram_mib", lambda: 6_690)
    monkeypatch.setattr(
        qm, "gpu_holders", lambda: [{"pid": "12345", "name": "python", "used_mib": "17214"}]
    )

    with pytest.raises(ModelUnavailable) as caught:
        server.model.ready()

    assert "6690 MiB" in str(caught.value)
    assert "kill 12345" in str(caught.value)
    assert server.spawned == []


def test_ready_refuses_a_load_that_did_not_reach_the_gpu(server, monkeypatch, tmp_path):
    """A CPU fallback loads fine and takes almost no VRAM -- only the clock tells."""
    big = tmp_path / "big.gguf"
    with big.open("wb") as f:
        f.truncate(4 * 2**30)  # 4 GiB, sparse
    server.model.model_path = str(big)
    frees = iter([24_000, 23_700])
    monkeypatch.setattr(qm, "free_vram_mib", lambda: next(frees))

    with pytest.raises(ModelUnavailable, match="not on the GPU"):
        server.model.ready()
    assert server.spawned[0].terminated


def test_a_server_that_dies_while_loading_is_reported_with_its_log(server, monkeypatch):
    def spawn(self, args, log, env):
        log.write("CUDA error: out of memory\n")
        proc = FakeProc()
        proc.returncode = 1
        return proc

    monkeypatch.setattr(QwenModel, "_spawn", spawn)

    with pytest.raises(ModelUnavailable) as caught:
        server.model.ready()

    assert "exited with code 1" in str(caught.value)
    assert "out of memory" in str(caught.value)


def test_ready_waits_out_the_servers_loading_answer(server, monkeypatch):
    """The real server answers /health with 503 "Loading model" for ~10s."""
    loading = iter([True, True, False])
    original = server.request

    def request(path, body=None, timeout=None):
        if path == "/health" and next(loading):
            raise qm.ServerError("llama-server /health -> 503: Loading model")
        return original(path, body, timeout)

    monkeypatch.setattr(QwenModel, "_request", lambda self, *a, **k: request(*a, **k))
    monkeypatch.setattr(qm.time, "sleep", lambda s: None)

    server.model.ready()

    assert len(server.spawned) == 1


def test_close_stops_the_server(server):
    server.model.ready()
    server.model.close()
    server.model.close()  # idempotent

    assert server.spawned[0].terminated


# --- generation -------------------------------------------------------------


def test_generate_sends_the_rendered_prompt_and_sampling_settings(server):
    g = server.model.generate(Prompt(system="sys", user="label text", stop=("\nEND",), max_tokens=400))

    body = server.bodies[-1]
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


def test_a_schema_is_sent_as_the_servers_json_schema(server):
    schema = {"type": "object"}

    server.model.generate(Prompt(system="s", user="u", schema=schema))

    assert server.bodies[-1]["json_schema"] == schema


def test_submit_keeps_every_slot_busy_and_no_more(server):
    server.model.ready()
    server.delay_s = 0.1

    started = time.monotonic()
    futures = [server.model.submit(Prompt(system="s", user=f"label {i}")) for i in range(8)]
    out = [f.result() for f in futures]

    assert len(out) == 8
    assert server.in_flight_peak == 4  # n_parallel, never more
    assert time.monotonic() - started < 0.35  # two rounds of 0.1s, not eight


def test_close_cancels_what_is_still_queued(server):
    server.model.ready()
    server.delay_s = 0.2
    futures = [server.model.submit(Prompt(system="s", user=f"label {i}")) for i in range(8)]

    server.model.close()

    assert any(f.cancelled() for f in futures)


# --- planning ---------------------------------------------------------------


def prompt_of(n_words: int, max_tokens: int = 400) -> Prompt:
    return Prompt(system="s", user=" ".join(["w"] * n_words), max_tokens=max_tokens)


def test_plan_keeps_the_given_order(server):
    """Longest-first was measured worse: it defeated the system-prompt cache."""
    prompts = [prompt_of(n) for n in (10, 50, 30, 20, 40, 60)]

    assert server.model.plan(prompts) == Plan(order=(0, 1, 2, 3, 4, 5), too_long=())


def test_plan_sets_aside_what_cannot_fit_a_slot(server):
    """Prompt + output budget + repair margin must fit, or it is never sent."""
    fits = prompt_of(100, max_tokens=400)
    overflows = prompt_of(4800, max_tokens=400)  # 4800 + 400 + 64 > 5120

    plan = server.model.plan([fits, overflows])

    assert plan.order == (0,)
    assert plan.too_long == (1,)


def test_the_output_budget_counts_toward_the_fit(server):
    short_answer = prompt_of(3100, max_tokens=400)
    long_answer = prompt_of(3100, max_tokens=2000)  # 3100 + 2000 + 64 > 5120

    assert server.model.plan([short_answer, long_answer]).too_long == (1,)


# --- priming ----------------------------------------------------------------


def test_the_shared_prefix_is_cut_after_the_last_full_turn():
    rendered = [
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nlabel one",
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nlabel two",
    ]

    assert qm.shared_prefix(rendered) == "<|im_start|>system\nS<|im_end|>\n"


def test_nothing_is_shared_by_a_single_prompt_or_different_systems():
    assert qm.shared_prefix(["<|im_start|>system\nS<|im_end|>\nx"]) == ""
    assert qm.shared_prefix(["<|im_start|>system\nA<|im_end|>\n", "<|im_start|>system\nB<|im_end|>\n"]) == ""


def test_plan_primes_every_slot_with_the_shared_system_turn(server):
    prompts = [Prompt(system="sys", user=f"label {i}") for i in range(6)]
    server.model.ready()
    before = len(server.bodies)

    server.model.plan(prompts)

    primes = server.bodies[before:]
    assert len(primes) == 4  # one per slot
    assert all(b["prompt"] == "<|im_start|>system\nsys<|im_end|>\n" for b in primes)
    assert all(b["n_predict"] == 1 for b in primes)


def test_a_second_plan_for_the_same_prefix_does_not_prime_again(server):
    prompts = [Prompt(system="sys", user=f"label {i}") for i in range(6)]
    server.model.plan(prompts)
    before = len(server.bodies)

    server.model.plan(prompts[:3])

    assert len(server.bodies) == before


def test_a_restarted_server_is_primed_afresh(server):
    prompts = [Prompt(system="sys", user=f"label {i}") for i in range(6)]
    server.model.plan(prompts)
    server.model.close()
    server.model.ready()
    before = len(server.bodies)

    server.model.plan(prompts)

    assert len(server.bodies) - before == 4
