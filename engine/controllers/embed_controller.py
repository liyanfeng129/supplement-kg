"""Drives one `EmbedModel`: readies it, batches texts into requests, reports progress.

**Per model, not per workflow.** This controller imports `EmbedModel`
concretely and knows nothing about what the texts are: no label, no passage.
The instruction is a string the caller hands in; where it came from is the
caller's business.

**The model is the bottleneck; its queue must never be empty while work
remains.** `stream` pulls texts lazily, groups them in pull order into requests
of up to `batch`, and keeps `in_flight` requests outstanding. More texts must be
queued than the model runs at once, so it is never left idle: `in_flight x
batch` exceeds that number. The controller never learns how many the model
runs at once (servers, slots: the model's business); the ratio is kept by hand
in `configs/embed.toml`.

**One instruction per controller.** Vectors made under different instructions
are not comparable, so it is fixed at construction and passed on every `fits`
and `submit`.

**Fit is checked per text, before batching.** A text the model says cannot run
in one pass is handed back at once as `fits_context` and never reaches the
model. Never truncated.

**A failed request is re-sent whole**, until `max_attempts`; then each of its
texts settles as `error` and the stream goes on. Only `ModelUnavailable` -- the
model says it cannot run at all -- ends it. There are no checks and no parser:
an embedding has no text to validate, and shape and sanity are the model's to
guarantee, in `ready()`.

**The controller creates and owns its model.** `EmbedController()` builds
`EmbedModel()`, and `close()` closes it. Passing a model in is the seam tests
use.

**Progress is counted here, per text**, as Qwen's controller counts it per
prompt: done of total, tokens per second over the wall clock, and an ETA over
the last `_ETA_WINDOW` texts. Nothing is printed; it is relayed to
`on_progress` and the log.
"""

from __future__ import annotations

import collections
import itertools
import logging
import time
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, wait
from dataclasses import dataclass
from typing import Hashable, Iterable, Iterator, Sequence, TypeVar

import numpy as np

from engine import config
from engine.models.base_model import BaseEmbedModel, ModelUnavailable, Progress, ProgressFn
from engine.models.embed_model import EmbedModel

log = logging.getLogger(__name__)

# Used where `configs/embed.toml` has no `[controller]` value, or a non-integer one.
_FALLBACKS = {"batch": 32, "in_flight": 4, "max_attempts": 3}

# The failure name given to a text the model says cannot run in one pass.
TOO_LONG = "fits_context"

# The failure name given to a text whose request failed on every attempt.
ERROR = "error"

# Texts the ETA is measured over: at tens of texts a second, a few minutes.
_ETA_WINDOW = 10_000

# The caller's key for one text. Opaque to the controller.
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True)
class Outcome:
    """The end state of one text, however many attempts its request took.

    `failed` is one name, not a list: an embedding fails at most one way.
    """

    vector: np.ndarray | None  # float32, (dim,); None unless ok
    ok: bool
    attempts: int
    failed: str | None  # TOO_LONG or ERROR; None when ok


class _Meter:
    """Accumulates one run's facts into the `Progress` it reports. Copied from Qwen's.

    The clock starts when the meter is built -- after the model is ready -- so
    a load never inflates the rate. Without a total there is no ETA.

    `total` and `done` are this session's; `n_already_done` is what earlier
    sessions of the same run finished, added to both for the reported
    progress. Tokens per second is over the whole session's wall clock; the
    ETA is from the last `_ETA_WINDOW` texts, so a slowdown shows quickly.
    """

    def __init__(self, total: int | None, n_already_done: int = 0):
        self.total = total
        self.n_already_done = n_already_done
        self.done = 0
        self.too_long = 0
        self.errors = 0
        self.tokens = 0
        self.started = time.monotonic()
        # The time at the start and after each text; one more entry than the
        # window, so it spans `_ETA_WINDOW` texts.
        self.recent = collections.deque([self.started], maxlen=_ETA_WINDOW + 1)

    def text_done(self) -> Progress:
        def duration(seconds: float) -> str:
            seconds = int(round(seconds))
            if seconds < 60:
                return f"{seconds}s"
            if seconds < 3600:
                return f"{seconds // 60}m{seconds % 60:02d}s"
            return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"

        self.done += 1
        now = time.monotonic()
        self.recent.append(now)
        elapsed = now - self.started
        span, units = self.recent[-1] - self.recent[0], len(self.recent) - 1
        parts = []
        if self.n_already_done:
            of = "" if self.total is None else f"/{self.total}"
            parts.append(f"(this session {self.done}{of})")
        parts.append(f"{self.tokens} tok")
        if elapsed > 0:
            parts.append(f"{self.tokens / elapsed:.0f} tok/s")
        if self.too_long:
            parts.append(f"{self.too_long} too long")
        if self.errors:
            parts.append(f"{self.errors} failed")
        remaining = None if self.total is None else self.total - self.done
        if remaining and span > 0:
            parts.append(f"eta {duration(span / units * remaining)}")
        return Progress(
            done=self.n_already_done + self.done,
            total=None if self.total is None else self.n_already_done + self.total,
            detail=" · ".join(parts),
        )


class EmbedController:
    """Owns one embedding model and runs texts through it to vectors."""

    def __init__(
        self,
        model: BaseEmbedModel | None = None,
        instruction: str | None = None,
        batch: int | None = None,
        in_flight: int | None = None,
        max_attempts: int | None = None,
    ):
        """Creates `EmbedModel()` unless given one. Batching, in-flight count and budget from configs/embed.toml unless passed."""
        self.model = EmbedModel() if model is None else model
        self.instruction = instruction

        # `[controller]` from the file, each value falling back on its own when
        # missing or not an integer: a broken file must not raise here.
        table = config.read_toml(config.EMBED_CONFIG_PATH).get("controller")
        table = table if isinstance(table, dict) else {}
        passed = {"batch": batch, "in_flight": in_flight, "max_attempts": max_attempts}
        for name, fallback in _FALLBACKS.items():
            value = table.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                value = fallback
            value = value if passed[name] is None else passed[name]
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
            setattr(self, name, value)

    def close(self) -> None:
        """Close the model: stop its server, free the card. Idempotent."""
        self.model.close()

    def stream(
        self,
        work: Iterable[tuple[K, str]],
        total: int | None = None,
        n_already_done: int = 0,
        on_progress: ProgressFn | None = None,
    ) -> Iterator[tuple[K, Outcome]]:
        """Embed each `(key, text)`, yielding `(key, outcome)` as each settles.

        `work` is pulled lazily: at most `in_flight` requests of up to `batch`
        texts are outstanding, and settling one makes room to pull and submit
        the next. Outcomes come out in completion order; the key is what maps
        them back. `total`, when given, is the denominator for progress and the
        ETA: texts in this `work`. `n_already_done` is what earlier sessions of
        a resumed run finished, counted into the reported progress.

        An empty text raises `ValueError`: it is a caller bug, and one that
        reached the server would fail its whole batch with a misleading reason.

        Stopping early (a `break`, an exception, Ctrl-C) cancels the requests
        still queued; their texts have no outcome, so a resumed run re-runs
        them.
        """

        def report(progress: Progress) -> None:
            of = "" if progress.total is None else f"/{progress.total}"
            log.info("%d%s %s", progress.done, of, progress.detail)
            if on_progress:
                on_progress(progress)

        work = iter(work)

        # 1. Ready the model, unless there is nothing to embed: a load for zero
        # texts is pure cost.
        first = next(work, None)
        if first is None:
            return
        work = itertools.chain([first], work)
        try:
            self.model.ready()
        except ModelUnavailable as exc:
            which = "the" if total is None else f"the {total}"
            raise ModelUnavailable(
                f"The model could not be readied, so none of {which} texts were "
                f"embedded.\n\n{exc}"
            ) from exc

        meter = _Meter(total, n_already_done)
        # future -> (the request's (key, text) pairs, attempts so far). Insertion
        # order is submit order, so settling the oldest finished future first
        # keeps a run deterministic when several finish together.
        outstanding: dict[Future, tuple[list[tuple[K, str]], int]] = {}
        exhausted = False
        try:
            while True:
                # 2. Top up to `in_flight` requests. Each is filled, in pull
                # order, with up to `batch` texts that fit; one that does not
                # fit is handed back at once and never sent.
                while not exhausted and len(outstanding) < self.in_flight:
                    request: list[tuple[K, str]] = []
                    while len(request) < self.batch:
                        item = next(work, None)
                        if item is None:
                            exhausted = True
                            break
                        key, text = item
                        if not text:
                            raise ValueError(
                                f"text {key!r} is empty: nothing to embed, and sent it would "
                                "fail its whole batch"
                            )
                        if not self.model.fits(text, self.instruction):
                            log.warning("text %r does not fit one pass of the model; not sent", key)
                            meter.too_long += 1
                            report(meter.text_done())
                            yield key, Outcome(vector=None, ok=False, attempts=0, failed=TOO_LONG)
                            continue
                        request.append((key, text))
                    if request:
                        future = self.model.submit([text for _, text in request], self.instruction)
                        outstanding[future] = (request, 0)

                # 3. Done when the work is used up and nothing is outstanding.
                if not outstanding:
                    return

                # 4. Settle the oldest finished request: retry it, or hand its
                # texts back.
                done, _ = wait(outstanding, return_when=FIRST_COMPLETED)
                future = next(f for f in outstanding if f in done)
                request, n = outstanding.pop(future)
                n += 1
                try:
                    embedding = future.result()
                except (ModelUnavailable, CancelledError):
                    raise
                except Exception as exc:
                    # The request failed; the model says the server is still
                    # fit, or it would have raised ModelUnavailable. Re-send it
                    # as it was, as a whole batch.
                    if n < self.max_attempts:
                        log.info(
                            "request of %d texts, attempt %d/%d failed with %r; retrying",
                            len(request), n, self.max_attempts, exc,
                        )
                        retry = self.model.submit([text for _, text in request], self.instruction)
                        outstanding[retry] = (request, n)
                        continue
                    log.warning(
                        "request of %d texts: giving up after %d attempts; last error %r",
                        len(request), n, exc,
                    )
                    for key, _ in request:
                        meter.errors += 1
                        report(meter.text_done())
                        yield key, Outcome(vector=None, ok=False, attempts=n, failed=ERROR)
                    continue

                meter.tokens += embedding.usage.prompt_tokens
                for (key, _), row in zip(request, embedding.vectors):
                    report(meter.text_done())
                    yield key, Outcome(vector=row, ok=True, attempts=n, failed=None)
        finally:
            for future in outstanding:
                future.cancel()

    def handle(self, texts: Sequence[str], on_progress: ProgressFn | None = None) -> np.ndarray:
        """Embed every text; one `(n, dim)` array, rows in input order.

        `stream` over a list already in memory, for tests, notebooks and small
        sets. Blocking. If any text is not `ok`, raises `ValueError` naming
        the failed indices and why: an array has no place for a missing row.
        No texts give an empty `(0, 0)` array, and the model is not readied.
        """
        texts = list(texts)
        outcomes: list[Outcome | None] = [None] * len(texts)
        for i, outcome in self.stream(enumerate(texts), total=len(texts), on_progress=on_progress):
            outcomes[i] = outcome
        failed = [(i, o.failed) for i, o in enumerate(outcomes) if o is not None and not o.ok]
        if failed:
            listed = ", ".join(f"{i}: {why}" for i, why in failed)
            raise ValueError(f"{len(failed)} of {len(texts)} texts were not embedded -- {listed}")
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([o.vector for o in outcomes])  # type: ignore[union-attr]
