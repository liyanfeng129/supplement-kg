"""EmbedController batches texts into requests, keeps the model fed, retries failures.

What is pinned: `handle` returns one row per text in input order, however the
requests settle; `stream` pulls work lazily and keeps at most `in_flight`
requests outstanding; a failed request is re-sent whole up to `max_attempts`,
then its texts settle as `error` and the stream goes on; `ModelUnavailable`
ends the stream; a text that does not fit settles as `fits_context` and never
reaches the model; an empty text is refused; the controller's instruction
reaches every `fits` and `submit`; progress reports done of total, tokens per
second and an ETA.

The model here is a stand-in -- no GPU, no server. It is a `BaseEmbedModel`,
not an `EmbedModel` subclass: `EmbedModel()` is one shared instance.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from engine.controllers import embed_controller as ec
from engine.controllers.embed_controller import EmbedController, Outcome
from engine.models.base_model import BaseEmbedModel, Embedding, ModelUnavailable, Usage


def vector(text: str) -> np.ndarray:
    """Text `t7` embeds as [7, 0, 0]: a row a test can recognise."""
    return np.array([float(text[1:]), 0.0, 0.0], dtype=np.float32)


class FakeEmbed(BaseEmbedModel):
    """Embeds each request on a worker thread; records what it saw.

    `delay(texts)` is how long a request takes, so a test decides which
    settles first. `failures[text]` is how many times a request holding that
    text raises before it succeeds; an exception instance is raised as it is.
    A text in `too_long` does not fit.
    """

    def __init__(self, delay=lambda texts: 0.0, failures=None, too_long=(), workers=8):
        self.delay = delay
        self.failures = dict(failures or {})
        self.too_long = set(too_long)
        self.requests: list[list[str]] = []
        self.instructions: list[tuple[str, str | None]] = []
        self.pending = 0
        self.pending_peak = 0
        self.closed = 0
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def close(self) -> None:
        self.closed += 1

    def fits(self, text, instruction=None):
        self.instructions.append(("fits", instruction))
        return text not in self.too_long

    def embed(self, texts, instruction=None):
        time.sleep(self.delay(texts))
        with self._lock:
            self.requests.append(list(texts))
            for text in texts:
                failure = self.failures.get(text)
                if isinstance(failure, BaseException):
                    raise failure
                if failure:
                    self.failures[text] -= 1
                    raise RuntimeError(f"request with {text} failed")
        return Embedding(
            vectors=np.stack([vector(t) for t in texts]),
            latency_ms=10,
            usage=Usage(prompt_tokens=100 * len(texts)),
        )

    def submit(self, texts, instruction=None):
        self.instructions.append(("submit", instruction))
        with self._lock:
            self.pending += 1
            self.pending_peak = max(self.pending_peak, self.pending)

        def settled(_):
            with self._lock:
                self.pending -= 1

        future = self._pool.submit(self.embed, list(texts), instruction)
        future.add_done_callback(settled)
        return future


def texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


def controller(model, **kwargs) -> EmbedController:
    settings = {"batch": 2, "in_flight": 2, "max_attempts": 3}
    settings.update(kwargs)
    return EmbedController(model, **settings)


# --- construction -----------------------------------------------------------


def test_the_committed_file_supplies_the_controller_settings():
    c = EmbedController(FakeEmbed())

    assert (c.batch, c.in_flight, c.max_attempts) == (16, 2, 3)
    assert c.instruction is None


def test_close_closes_the_model():
    model = FakeEmbed()

    EmbedController(model).close()

    assert model.closed == 1


# --- order, batching and the in-flight bound ---------------------------------


def test_handle_returns_vectors_in_input_order_though_batches_settle_out_of_order():
    # The first batch is the slowest, the last the fastest.
    model = FakeEmbed(delay=lambda batch: 0.1 if batch[0] == "t0" else 0.0)

    vectors = controller(model, batch=2, in_flight=3).handle(texts(5))

    assert vectors.shape == (5, 3)
    assert list(vectors[:, 0]) == [0, 1, 2, 3, 4]
    assert model.requests[-1] == ["t0", "t1"]  # it really did settle last


def test_texts_are_grouped_in_pull_order_into_requests_of_up_to_batch():
    model = FakeEmbed()

    controller(model, batch=2, in_flight=1).handle(texts(5))

    assert model.requests == [["t0", "t1"], ["t2", "t3"], ["t4"]]


def test_stream_keeps_at_most_in_flight_requests_and_pulls_work_lazily():
    model = FakeEmbed(delay=lambda batch: 0.02)
    pulled = []

    def work():
        for i in range(40):
            pulled.append(i)
            yield i, f"t{i}"

    stream = controller(model, batch=3, in_flight=2).stream(work())
    next(stream)
    pulled_at_first_outcome = len(pulled)
    rest = list(stream)

    assert pulled_at_first_outcome <= 2 * 3
    assert len(rest) == 39
    assert model.pending_peak == 2


def test_stream_yields_each_key_with_its_vector():
    model = FakeEmbed()

    out = dict(controller(model).stream([("a", "t1"), ("b", "t2")]))

    assert out["a"].ok and out["a"].attempts == 1 and out["a"].failed is None
    assert out["b"].vector.tolist() == [2.0, 0.0, 0.0]


# --- retry and stop ---------------------------------------------------------


def test_a_failed_request_is_re_sent_whole_and_every_text_carries_its_attempts():
    model = FakeEmbed(failures={"t1": 1})

    out = dict(controller(model, batch=2).stream(enumerate(texts(4))))

    assert model.requests.count(["t0", "t1"]) == 2
    assert [out[i].attempts for i in range(4)] == [2, 2, 1, 1]
    assert all(o.ok for o in out.values())


def test_a_request_that_fails_every_attempt_settles_as_error_and_the_stream_goes_on():
    model = FakeEmbed(failures={"t1": 99})

    out = dict(controller(model, batch=2, max_attempts=3).stream(enumerate(texts(4))))

    for i in (0, 1):
        assert out[i] == Outcome(vector=None, ok=False, attempts=3, failed=ec.ERROR)
    assert out[2].ok and out[3].ok
    assert model.requests.count(["t0", "t1"]) == 3


def test_model_unavailable_ends_the_stream():
    model = FakeEmbed(failures={"t2": ModelUnavailable("the server is gone")})

    with pytest.raises(ModelUnavailable, match="gone"):
        list(controller(model, batch=1, in_flight=1).stream(enumerate(texts(5))))

    assert ["t3"] not in model.requests


def test_handle_names_the_texts_that_failed():
    model = FakeEmbed(failures={"t1": 99}, too_long={"t3"})

    with pytest.raises(ValueError, match=r"1: error.*3: fits_context|3: fits_context.*1: error"):
        controller(model, batch=1).handle(texts(4))


# --- fit, empty text, instruction -------------------------------------------


def test_a_text_that_does_not_fit_settles_as_too_long_and_never_reaches_the_model():
    model = FakeEmbed(too_long={"t1"})

    out = dict(controller(model).stream(enumerate(texts(3))))

    assert out[1] == Outcome(vector=None, ok=False, attempts=0, failed=ec.TOO_LONG)
    assert all("t1" not in request for request in model.requests)
    assert out[0].ok and out[2].ok


def test_an_empty_text_is_refused():
    with pytest.raises(ValueError, match="empty"):
        controller(FakeEmbed()).handle(["t0", ""])


def test_the_instruction_reaches_every_fits_and_submit():
    model = FakeEmbed()

    controller(model, instruction="Find the product", batch=2).handle(texts(5))

    assert {verb for verb, _ in model.instructions} == {"fits", "submit"}
    assert {instruction for _, instruction in model.instructions} == {"Find the product"}


def test_handle_of_no_texts_returns_an_empty_array_without_touching_the_model():
    model = FakeEmbed()

    vectors = controller(model).handle([])

    assert vectors.shape[0] == 0
    assert model.instructions == []


# --- progress ---------------------------------------------------------------


def test_progress_reports_done_of_total_tokens_per_second_and_an_eta(monkeypatch):
    clock = iter(range(1000))
    monkeypatch.setattr(ec.time, "monotonic", lambda: float(next(clock)))
    model = FakeEmbed()
    seen = []

    controller(model, batch=2, in_flight=1).handle(texts(4), on_progress=seen.append)

    assert [(p.done, p.total) for p in seen] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert "tok/s" in seen[0].detail
    assert "eta" in seen[0].detail
    assert "eta" not in seen[-1].detail


def test_a_resumed_run_counts_what_earlier_sessions_did():
    seen = []

    list(
        controller(FakeEmbed()).stream(
            enumerate(texts(2)), total=2, n_already_done=10, on_progress=seen.append
        )
    )

    assert (seen[-1].done, seen[-1].total) == (12, 12)
    assert "this session 2/2" in seen[-1].detail
