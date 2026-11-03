"""CodexController runs every prompt to an outcome, keeping the model fed, retrying failures.

What is pinned: `stream` pulls work lazily and keeps at most `in_flight`
prompts outstanding, handing each outcome back with the caller's key as it
settles; it re-submits only failures -- repaired with the failed checks'
messages -- up to the budget; a prompt that never passes comes back
`ok=False` with the reasons and does not stall the run; a prompt the model
says does not fit is reported, never sent; stopping early cancels what is
queued. `handle` is `stream` over a list, with outcomes back in input order;
progress counts units, not attempts; and a model that cannot be readied stops
the run before anything is sent.

The model here is a stand-in that answers from a script -- no CLI runs.
The file is a copy of `test_qwen_controller.py`, as the controller is a copy
of `QwenController` (spec 0012).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from engine.controllers import codex_controller as qc
from engine.controllers.codex_controller import Outcome, CodexController
from engine.models.base_model import (
    BaseModel,
    Check,
    Generation,
    ModelUnavailable,
    Prompt,
    Usage,
)
from engine import config


class ScriptedCodex(BaseModel):
    """Answers each prompt with the next scripted text; records what it saw.

    Answers each `submit()` before returning (the base model's default), so
    the script is consumed in submit order. A reply that is an exception is
    raised instead, as a failed request. A prompt whose user text is in
    `too_long` does not fit.
    """

    submit = BaseModel.submit

    def __init__(self, replies, fail_ready=None, too_long=()):
        self.replies = list(replies)
        self.fail_ready = fail_ready
        self.too_long = set(too_long)
        self.seen: list[Prompt] = []
        self.samples: list[list[Prompt]] = []
        self.readied = 0
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def ready(self) -> None:
        self.readied += 1
        if self.fail_ready:
            raise ModelUnavailable(self.fail_ready)

    def fits(self, prompt: Prompt) -> bool:
        return prompt.user not in self.too_long

    def cache_shared_prefix(self, sample) -> None:
        self.samples.append(list(sample))

    def generate(self, prompt: Prompt) -> Generation:
        self.seen.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return Generation(
            text=reply,
            latency_ms=1000,
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            stop_reason="stop",
        )


is_json = Check(
    name="is_json",
    passes=lambda _prompt, generation: generation.text.startswith("{"),
    message="Answer in JSON.",
)

has_claims = Check(
    name="has_claims",
    passes=lambda _prompt, generation: "claim" in generation.text,
    message="Include a claim.",
)


def prompt(user: str = "extract") -> Prompt:
    return Prompt(system="sys", user=user)


# --- the retry loop ---------------------------------------------------------


def test_a_passing_first_answer_is_one_attempt():
    model = ScriptedCodex(['{"claim": 1}'])

    [outcome] = CodexController(model).handle([prompt()], checks=[is_json])

    assert outcome == Outcome(
        generation=model_generation('{"claim": 1}'), ok=True, attempts=1, failed=[]
    )


def test_a_failed_check_s_message_is_in_the_re_sent_prompt():
    model = ScriptedCodex(["no json here", '{"claim": 1}'])

    [outcome] = CodexController(model).handle([prompt("extract")], checks=[is_json])

    assert outcome.ok and outcome.attempts == 2
    retry = model.seen[1]
    assert retry.user.startswith("extract")
    assert "- Answer in JSON." in retry.user
    assert "is_json" not in retry.user  # the name is for people, not the model
    assert retry.system == "sys"


def test_the_repair_carries_only_the_checks_that_failed():
    model = ScriptedCodex(['{"x": 1}', '{"claim": 1}'])

    CodexController(model).handle([prompt()], checks=[is_json, has_claims])

    assert has_claims.message in model.seen[1].user
    assert is_json.message not in model.seen[1].user


def test_each_repair_starts_from_the_original_prompt():
    """Repairs must not pile up: attempt 3 carries attempt 2's failures only."""
    model = ScriptedCodex(["plain", '{"x": 1}', '{"claim": 1}'])

    CodexController(model).handle([prompt()], checks=[is_json, has_claims])

    third = model.seen[2].user
    assert has_claims.message in third
    assert is_json.message not in third


def test_it_gives_up_after_the_budget_and_says_why():
    model = ScriptedCodex(["a", "b", "c"])

    [outcome] = CodexController(model, max_attempts=3).handle(
        [prompt()], checks=[is_json]
    )

    assert not outcome.ok
    assert outcome.attempts == 3
    assert outcome.failed == ["is_json"]
    assert outcome.generation.text == "c"
    assert len(model.seen) == 3


def test_one_unsatisfiable_prompt_does_not_stall_the_batch():
    # Retries queue behind first attempts: one#1, two#1, one#2.
    model = ScriptedCodex(["bad", '{"claim": 2}', "bad"])

    outcomes = CodexController(model, max_attempts=2).handle(
        [prompt("one"), prompt("two")], checks=[is_json]
    )

    assert [o.ok for o in outcomes] == [False, True]
    assert outcomes[1].generation.text == '{"claim": 2}'


def test_without_checks_every_answer_passes():
    model = ScriptedCodex(["anything"])

    [outcome] = CodexController(model).handle([prompt()])

    assert outcome.ok and outcome.attempts == 1


def test_a_check_sees_the_original_prompt_on_every_attempt():
    """A rule about the input must judge the input, not a repaired copy."""
    seen: list[str] = []
    grounded = Check(
        name="grounded",
        passes=lambda p, _generation: seen.append(p.user) or False,
        message="Copy from the label.",
    )
    model = ScriptedCodex(["x", "y"])

    CodexController(model, max_attempts=2).handle([prompt("the label")], checks=[grounded])

    assert seen == ["the label", "the label"]


def test_outcomes_line_up_with_prompts_by_position():
    model = ScriptedCodex(["first", "second", "third"])

    outcomes = CodexController(model).handle([prompt(), prompt(), prompt()])

    assert [o.generation.text for o in outcomes] == ["first", "second", "third"]


# --- submitting and fitting -------------------------------------------------


def test_prompts_are_submitted_in_input_order():
    model = ScriptedCodex(["x", "y", "z"])

    CodexController(model).handle([prompt("0"), prompt("1"), prompt("2")])

    assert [p.user for p in model.seen] == ["0", "1", "2"]


def test_only_failures_are_re_submitted():
    model = ScriptedCodex(['{"a": 1}', "bad", '{"b": 2}'])

    outcomes = CodexController(model).handle([prompt("one"), prompt("two")], checks=[is_json])

    assert [o.attempts for o in outcomes] == [1, 2]
    assert [p.user.split("\n")[0] for p in model.seen] == ["one", "two", "two"]


def test_a_prompt_that_does_not_fit_is_reported_not_sent():
    model = ScriptedCodex(["a", "c"], too_long=["1"])

    outcomes = CodexController(model).handle([prompt("0"), prompt("1"), prompt("2")])

    assert [o.ok for o in outcomes] == [True, False, True]
    assert outcomes[1].failed == [qc.TOO_LONG]
    assert outcomes[1].attempts == 0
    assert outcomes[1].generation.stop_reason == "too_long"
    assert [p.user for p in model.seen] == ["0", "2"]


def test_the_shared_prefix_is_cached_once_from_the_first_in_flight_prompts():
    model = ScriptedCodex(["a", "b", "c", "d", "e"])

    CodexController(model, in_flight=3).handle([prompt(str(i)) for i in range(5)])

    assert [[p.user for p in s] for s in model.samples] == [["0", "1", "2"]]


def test_outcomes_come_back_in_input_order_whatever_order_they_finish_in():
    """Acceptance 4: `handle` via `stream` still lines outcomes up by position."""
    model = SlowFirst(delays={"slow": 0.2, "fast": 0.0})

    outcomes = CodexController(model).handle([prompt("slow"), prompt("fast")])

    assert [o.generation.text for o in outcomes] == ["slow", "fast"]
    assert model.finished == ["fast", "slow"]


# --- streaming --------------------------------------------------------------


def test_stream_hands_back_the_callers_key_with_each_outcome():
    model = ScriptedCodex(["for a", "for b"])

    out = list(CodexController(model).stream([("id-a", prompt()), (("id", 2), prompt())]))

    assert [(key, o.generation.text) for key, o in out] == [("id-a", "for a"), (("id", 2), "for b")]


def test_stream_yields_in_completion_order():
    model = SlowFirst(delays={"slow": 0.2, "fast": 0.0})

    out = list(CodexController(model).stream([(1, prompt("slow")), (2, prompt("fast"))]))

    assert [key for key, _ in out] == [2, 1]


def test_stream_pulls_only_in_flight_prompts_before_the_first_outcome():
    """The rest of the work stays unread until a settled prompt makes room."""
    pulled = []

    def work():
        for i in range(10):
            pulled.append(i)
            yield i, prompt(str(i))

    stream = CodexController(ScriptedCodex(["x"] * 10), in_flight=3).stream(work())

    next(stream)
    assert pulled == [0, 1, 2]

    next(stream)
    assert pulled == [0, 1, 2, 3]  # one settled, one pulled to replace it

    assert len(list(stream)) == 8


def test_a_prompt_that_does_not_fit_is_replaced_at_once():
    """Never submitted, so it frees no slot and takes none: the next is pulled."""
    model = ScriptedCodex(["a", "b"], too_long=["long"])

    out = list(
        CodexController(model, in_flight=1).stream(
            [(0, prompt("short")), (1, prompt("long")), (2, prompt("other"))]
        )
    )

    assert {key: o.failed for key, o in out} == {0: [], 1: [qc.TOO_LONG], 2: []}
    assert [p.user for p in model.seen] == ["short", "other"]


def test_stopping_early_cancels_what_is_still_queued():
    """The consumer walks away: queued prompts must not keep the model busy."""
    model = SlowFirst(delays={}, workers=1, default_delay=0.05)

    stream = CodexController(model, in_flight=4).stream((i, prompt(str(i))) for i in range(8))
    next(stream)
    stream.close()

    assert any(f.cancelled() for f in model.futures)


def test_an_empty_stream_does_not_load_the_model():
    model = ScriptedCodex([])

    assert list(CodexController(model).stream(iter([]))) == []
    assert model.readied == 0


# --- in_flight --------------------------------------------------------------


def test_in_flight_comes_from_the_committed_file():
    assert CodexController(ScriptedCodex([])).in_flight == 8


def test_in_flight_from_a_missing_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CODEX_CONFIG_PATH", tmp_path / "absent.toml")

    assert CodexController(ScriptedCodex([])).in_flight == 4


def test_an_explicit_in_flight_wins():
    assert CodexController(ScriptedCodex([]), in_flight=2).in_flight == 2


def test_in_flight_below_one_is_refused():
    with pytest.raises(ValueError, match="in_flight"):
        CodexController(ScriptedCodex([]), in_flight=0)


# --- the budget -------------------------------------------------------------


def test_the_budget_comes_from_the_committed_file():
    assert CodexController(ScriptedCodex([])).max_attempts == 3


def test_a_budget_from_a_missing_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CODEX_CONFIG_PATH", tmp_path / "absent.toml")

    assert CodexController(ScriptedCodex([])).max_attempts == 3


def test_a_budget_that_is_not_an_integer_falls_back(monkeypatch, tmp_path):
    config_file = tmp_path / "codex.toml"
    config_file.write_text('[controller]\nmax_attempts = "3"\n')
    monkeypatch.setattr(config, "CODEX_CONFIG_PATH", config_file)

    assert CodexController(ScriptedCodex([])).max_attempts == 3


def test_an_explicit_budget_wins():
    assert CodexController(ScriptedCodex([]), max_attempts=5).max_attempts == 5


def test_a_budget_below_one_is_refused():
    with pytest.raises(ValueError, match="max_attempts"):
        CodexController(ScriptedCodex([]), max_attempts=0)


# --- readiness --------------------------------------------------------------


def test_the_model_is_readied_before_anything_is_sent():
    model = ScriptedCodex(["x"])

    CodexController(model).handle([prompt()])

    assert model.readied >= 1


def test_an_unready_model_stops_the_run_with_a_readable_reason():
    model = ScriptedCodex(["x"], fail_ready="Refusing to run: not logged in")

    with pytest.raises(ModelUnavailable) as caught:
        CodexController(model).handle([prompt(), prompt()])

    message = str(caught.value)
    assert "2 prompts" in message
    assert "not logged in" in message
    assert model.seen == []


def test_an_empty_batch_does_not_load_the_model():
    """A 40s load for zero prompts is pure cost."""
    model = ScriptedCodex([])

    assert CodexController(model).handle([]) == []
    assert model.readied == 0


# --- progress ---------------------------------------------------------------


def test_progress_counts_units_not_attempts():
    model = ScriptedCodex(["bad", '{"a": 1}', '{"b": 2}'])
    seen = []

    CodexController(model).handle(
        [prompt(), prompt()], checks=[is_json], on_progress=seen.append
    )

    assert [(p.done, p.total) for p in seen] == [(1, 2), (2, 2)]


def test_progress_detail_reports_tokens_and_what_is_left(monkeypatch):
    ticks = iter(range(100))  # the meter starts at 0; each unit done is 1s later
    monkeypatch.setattr(qc, "_now", lambda: float(next(ticks)))
    model = ScriptedCodex(["a", "b", "c", "d"])
    seen = []

    CodexController(model).handle([prompt()] * 4, on_progress=seen.append)

    first, last = seen[0].detail, seen[-1].detail
    assert "30 tok" in first  # one generation: 10 prompt + 20 completion
    assert "eta 3s" in first  # one second per unit so far, 3 units left
    assert "120 tok" in last
    assert "eta" not in last


def test_progress_without_a_total_reports_no_eta():
    seen = []

    list(CodexController(ScriptedCodex(["a", "b"])).stream(
        [(0, prompt()), (1, prompt())], on_progress=seen.append
    ))

    assert [(p.done, p.total) for p in seen] == [(1, None), (2, None)]
    assert all("eta" not in p.detail for p in seen)


def test_progress_keeps_a_running_count_of_prompts_that_do_not_fit():
    """Found one by one now, so a bad rate must show while the run is going."""
    seen = []

    CodexController(ScriptedCodex(["a"], too_long=["long"])).handle(
        [prompt("long"), prompt("short")], on_progress=seen.append
    )

    assert "1 too long" in seen[-1].detail


def test_a_resumed_run_reports_the_whole_run_and_this_session():
    seen = []

    list(CodexController(ScriptedCodex(["a", "b"])).stream(
        [(0, prompt()), (1, prompt())], total=2, n_already_done=40, on_progress=seen.append
    ))

    assert [(p.done, p.total) for p in seen] == [(41, 42), (42, 42)]
    assert "this session 1/2" in seen[0].detail


def test_the_eta_follows_the_recent_rate_not_the_whole_run_average(monkeypatch):
    """A slowdown must show within minutes, not be averaged away over days."""
    ticks = iter([0, 1, 2, 3, 4, 14, 24, 34, 44, 54, 64])  # the meter's start, then one per unit
    monkeypatch.setattr(qc, "_now", lambda: float(next(ticks)))
    monkeypatch.setattr(qc, "_ETA_WINDOW", 3)
    seen = []

    CodexController(ScriptedCodex(["x"] * 10)).handle([prompt()] * 10, on_progress=seen.append)

    # After 7 units at t=34: the last 3 took 30s, so 3 left is 30s. The
    # whole-run average (34s / 7) would say 15s.
    assert "eta 30s" in seen[6].detail


# --- owning the model -------------------------------------------------------


def test_without_a_model_the_controller_creates_the_codex_model():
    from engine.models.codex_model import CodexModel

    controller = CodexController()
    try:
        assert isinstance(controller.model, CodexModel)
    finally:
        controller.close()


def test_close_closes_the_model():
    model = ScriptedCodex([])

    CodexController(model).close()

    assert model.closed == 1


# --- failed requests --------------------------------------------------------


def test_a_failed_request_is_retried_as_it_was():
    """No repair note: the answer was never seen, so there is no rule to cite."""
    model = ScriptedCodex([RuntimeError("socket reset"), '{"claim": 1}'])

    [outcome] = CodexController(model).handle([prompt("extract")], checks=[is_json])

    assert outcome.ok and outcome.attempts == 2
    assert [p.user for p in model.seen] == ["extract", "extract"]


def test_a_request_that_keeps_failing_is_an_error_outcome_and_the_stream_goes_on():
    model = ScriptedCodex([RuntimeError("a"), RuntimeError("b"), "fine"])

    out = list(CodexController(model, max_attempts=2, in_flight=1).stream(
        [(0, prompt("one")), (1, prompt("two"))]
    ))

    (k0, failed), (k1, fine) = out
    assert (k0, failed.ok, failed.attempts, failed.failed) == (0, False, 2, [qc.ERROR])
    assert failed.generation.stop_reason == "error"
    assert (k1, fine.ok) == (1, True)


def test_model_unavailable_from_a_request_ends_the_stream():
    model = ScriptedCodex([ModelUnavailable("still rate limited after 3 waits"), "never"])

    with pytest.raises(ModelUnavailable, match="rate limited"):
        list(CodexController(model).stream([(0, prompt()), (1, prompt())]))


# --- parsing ----------------------------------------------------------------


def test_ok_answers_are_parsed_and_failed_ones_are_not():
    model = ScriptedCodex(['{"claim": 1}', "bad", "bad"], too_long=["long"])
    parse = lambda text: {"text": text}  # noqa: E731

    out = dict(CodexController(model, max_attempts=2).stream(
        [(0, prompt("one")), (1, prompt("two")), (2, prompt("long"))], [is_json], parse
    ))

    assert out[0].parsed == {"text": '{"claim": 1}'}
    assert out[1].parsed is None and not out[1].ok
    assert out[2].parsed is None


def test_without_a_parser_nothing_is_parsed():
    [outcome] = CodexController(ScriptedCodex(["x"])).handle([prompt()])

    assert outcome.parsed is None


# --- helpers ----------------------------------------------------------------


class SlowFirst(BaseModel):
    """Runs prompts on real worker threads, each sleeping as long as its user text says.

    Answers with the user text, so an outcome names its prompt. Keeps every
    future it hands out and the order generations finished in.
    """

    def __init__(self, delays, workers=2, default_delay=0.0):
        self.delays = delays
        self.default_delay = default_delay
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures = []
        self.finished: list[str] = []
        self._lock = threading.Lock()

    def generate(self, prompt: Prompt) -> Generation:
        time.sleep(self.delays.get(prompt.user, self.default_delay))
        with self._lock:
            self.finished.append(prompt.user)
        return Generation(text=prompt.user, latency_ms=1)

    def submit(self, prompt: Prompt):
        future = self.pool.submit(self.generate, prompt)
        self.futures.append(future)
        return future


def model_generation(text: str) -> Generation:
    return Generation(
        text=text,
        latency_ms=1000,
        usage=Usage(prompt_tokens=10, completion_tokens=20),
        stop_reason="stop",
    )
