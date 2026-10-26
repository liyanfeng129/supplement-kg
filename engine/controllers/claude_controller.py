"""Drives one `ClaudeModel`: readies it, runs prompts to outcomes, reports progress.

**Per model, not per workflow.** This controller knows exactly which model it
drives -- it imports `ClaudeModel` concretely, the one-controller-per-model rule
made literal -- and nothing about what the prompts are for. A copy of
`QwenController`, not a shared base: only the model differs (spec 0012), and a
shared controller waits until the copies have stayed identical.

**Claude is a CLI call, not a slot, and that is the model's business.** The
model never says a prompt does not fit, so a `TOO_LONG` outcome never occurs;
it waits out rate limits itself, so one slow attempt here may be several
calls there; and it raises `ModelUnavailable` on a quota or a run of failures,
which ends the stream as any `ModelUnavailable` does.

**The model is the bottleneck; its queue must never be empty while work
remains.** `stream` pulls prompts lazily from the caller and keeps a fixed
number of them outstanding -- `in_flight`, more than the model runs at once --
so a slot that frees always finds a prompt already queued. Each settled
prompt hands back its outcome and makes room for one more pull. How many run
at once is the model's business: the controller never learns a slot count,
and never waits on a batch -- a slot that frees is refilled at once, which
measured 1.38x over one at a time where batch-by-batch measured 1.01x.

**One queue, and it is the model's.** A submitted prompt lives in the model's
pool. The controller keeps only a record per outstanding prompt -- the caller's
key, the prompt (a retry is built from it) and its attempts so far -- and
`in_flight` bounds how many records it keeps, which is how it bounds what it
pulls.

**Fit is checked per prompt, just before submitting.** The model says whether
a prompt fits a slot; one that does not is handed back at once as a
`fits_context` outcome and never reaches the model.

**Validation with retry lives here**, because the loop does. The checks come
with the prompt (see `repositories/prompts/`); the controller runs them
against each generation as it arrives and re-submits a failure at once --
repaired with each failed check's message -- until it passes or the budget is
spent. It can do that without understanding any check, because a check carries
its own message to the model. The prompt asks for a format and the controller
verifies it -- a prompt asking nicely is not a guarantee.

**A failed request is retried like a failed check**, within the same budget,
but re-sent as it was: no answer came back, so there is no rule to cite. One
that fails every attempt comes back as an `error` outcome and the stream goes
on. Only `ModelUnavailable` -- the model says it cannot run at all -- ends it.

**The parser comes with the prompt, like the checks.** An answer that passed
every check is parsed and the result carried in `Outcome.parsed`; the
controller runs the parser without knowing the format.

**The controller creates and owns its model.** `ClaudeController()` builds
`ClaudeModel()`, and `close()` closes it; a caller reaches the model only
through the controller. Passing a model in is the seam tests use.

**Progress is counted here, measured below.** The model reports tokens per
generation; the controller turns completed units and the wall clock into
done-of-total, throughput and an ETA. Wall clock rather than summed latency,
because concurrent generations overlap: four at 6s each is 6s, not 24. Rate
and ETA are over the last `_ETA_WINDOW` units, not the whole run, so a
slowdown shows within minutes; a resumed run counts what earlier sessions did
into done-of-total and reports this session's share beside it.

A check is a `Check` (see `models/base_model.py`): a name, a `passes(prompt,
generation)` test, and a message. It is run against the *original* prompt, so
a rule about the input sees the input, not a repaired copy. A check that raises
is a bug in the check and propagates.
"""

from __future__ import annotations

import collections
import itertools
import logging
import time
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Hashable, Iterable, Iterator, Sequence, TypeVar

from engine import config
from engine.models.base_model import (
    Check,
    Generation,
    ModelUnavailable,
    Progress,
    ProgressFn,
    Prompt,
)
from engine.models.claude_model import ClaudeModel

log = logging.getLogger(__name__)

# Used when `configs/claude.toml` has no `[controller].max_attempts`.
_FALLBACK_MAX_ATTEMPTS = 3

# Used when `configs/claude.toml` has no `[controller].in_flight`.
_FALLBACK_IN_FLIGHT = 4

# The failure name given to a prompt the model says cannot fit a slot.
TOO_LONG = "fits_context"

# The failure name given to a prompt whose request failed on every attempt.
ERROR = "error"

# Units the rate and ETA are measured over: at ~1s per unit, a few minutes.
_ETA_WINDOW = 200

# The caller's key for one unit of work. Opaque to the controller.
K = TypeVar("K", bound=Hashable)


def _now() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class Attempt:
    """One try at one prompt: what came back, and which checks it failed."""

    generation: Generation
    failed: list[str]  # check names, empty == passed


@dataclass(frozen=True)
class Outcome:
    """The end state of one unit of work, however many attempts it took.

    A bare `Generation` cannot say why a unit was given up on; this can.
    `stream` hands each back with the caller's key; `handle` returns them
    index-aligned with the prompts.
    """

    generation: Generation  # the last one, passing or not
    ok: bool
    attempts: int
    failed: list[str]  # names from the final attempt
    parsed: dict | None = None  # the parser's result; ok outcomes only


def load_max_attempts(path: Path | None = None) -> int:
    """`[controller].max_attempts` from the TOML, or the fallback.

    A value that is not an integer falls back like a missing one: a broken
    config file must not raise at construction.
    """
    table = config.read_toml(path or config.CLAUDE_CONFIG_PATH).get("controller")
    value = table.get("max_attempts") if isinstance(table, dict) else None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return _FALLBACK_MAX_ATTEMPTS


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


class _Meter:
    """Accumulates one run's facts into the `Progress` it reports.

    The clock starts when the meter is built -- after the model is ready and
    its cache is warm -- so a 40s load never inflates the ETA. Without a
    total there is no ETA, only done count and throughput.

    `total` and `done` are this session's; `n_already_done` is what earlier
    sessions of the same run finished, added to both for the reported
    progress. Throughput and the ETA come from the last `_ETA_WINDOW` units.
    """

    def __init__(self, total: int | None, n_already_done: int = 0):
        self.total = total
        self.n_already_done = n_already_done
        self.done = 0
        self.too_long = 0
        self.tokens = 0
        self.completion_tokens = 0
        self.started = _now()
        # (time, completion tokens so far) at the start and after each unit;
        # one more entry than the window, so it spans `_ETA_WINDOW` units.
        self.recent = collections.deque([(self.started, 0)], maxlen=_ETA_WINDOW + 1)

    def attempt(self, attempt: Attempt) -> None:
        usage = attempt.generation.usage
        self.tokens += usage.total_tokens
        self.completion_tokens += usage.completion_tokens

    def unit_done(self) -> Progress:
        self.done += 1
        self.recent.append((_now(), self.completion_tokens))
        (t0, tok0), (t1, tok1) = self.recent[0], self.recent[-1]
        span, units = t1 - t0, len(self.recent) - 1
        parts = []
        if self.n_already_done:
            of = "" if self.total is None else f"/{self.total}"
            parts.append(f"(this session {self.done}{of})")
        parts.append(f"{self.tokens} tok")
        if span > 0:
            parts.append(f"{(tok1 - tok0) / span:.1f} tok/s")
        if self.too_long:
            parts.append(f"{self.too_long} too long")
        remaining = None if self.total is None else self.total - self.done
        if remaining and span > 0:
            parts.append(f"eta {_duration(span / units * remaining)}")
        return Progress(
            done=self.n_already_done + self.done,
            total=None if self.total is None else self.n_already_done + self.total,
            detail=" · ".join(parts),
        )


class ClaudeController:
    """Owns one model and runs prompts through it to outcomes."""

    def __init__(
        self,
        model: ClaudeModel | None = None,
        max_attempts: int | None = None,
        in_flight: int | None = None,
    ):
        """Creates `ClaudeModel()` unless given one. Budget and in-flight count from configs/claude.toml unless overridden."""
        self.model = ClaudeModel() if model is None else model
        self.max_attempts = (
            load_max_attempts() if max_attempts is None else max_attempts
        )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")

        # `[controller].in_flight`, falling back like `max_attempts` does on a
        # missing or non-integer value.
        table = config.read_toml(config.CLAUDE_CONFIG_PATH).get("controller")
        value = table.get("in_flight") if isinstance(table, dict) else None
        if not isinstance(value, int) or isinstance(value, bool):
            value = _FALLBACK_IN_FLIGHT
        self.in_flight = value if in_flight is None else in_flight
        if self.in_flight < 1:
            raise ValueError(f"in_flight must be at least 1, got {self.in_flight}")

    # --- surface ------------------------------------------------------------

    def close(self) -> None:
        """Close the model: cancel what is queued, kill its running calls. Idempotent."""
        self.model.close()

    def stream(
        self,
        work: Iterable[tuple[K, Prompt]],
        checks: Iterable[Check] = (),
        parse: Callable[[str], dict] | None = None,
        total: int | None = None,
        n_already_done: int = 0,
        on_progress: ProgressFn | None = None,
    ) -> Iterator[tuple[K, Outcome]]:
        """Run each `(key, prompt)` to an outcome, yielding `(key, outcome)` as each settles.

        `work` is pulled lazily: at most `in_flight` prompts are outstanding,
        and one more is pulled each time one settles. Outcomes come out in
        completion order; the key is what maps them back. `total`, when
        given, is the denominator for progress and the ETA: prompts in this
        `work`. `n_already_done` is what earlier sessions of a resumed run
        finished, counted into the reported progress.

        `parse`, when given, is run on the text of every `ok` outcome and its
        result put in `Outcome.parsed`. A parser that raises is a bug and
        propagates, as a check that raises does.

        A request that fails is retried as it was, within `max_attempts`;
        after the last it comes back as an `ERROR` outcome and the stream goes
        on. `ModelUnavailable` ends the stream.

        Whatever the caller does between two `next()` calls runs on this
        loop's thread, while the model works through its queue -- keep it
        cheap. Stopping early (a `break`, an exception, Ctrl-C) cancels the
        prompts still queued; they have no outcome, so a resumed run re-runs
        them.
        """
        work = iter(work)
        checks = list(checks)

        # 1. Ready the model and warm its cache from the first prompts, once.
        first = list(itertools.islice(work, self.in_flight)) # slice off the first `in_flight` prompts to warm the cache and check fit.
        if not first:
            return
        self._ready(total)
        self.model.cache_shared_prefix([p for _, p in first]) 
        work = itertools.chain(first, work) 

        meter = _Meter(total, n_already_done)
        # future -> (key, original prompt, attempts so far). Insertion order is
        # submit order, so settling the oldest finished future first keeps a
        # run deterministic when several finish together.
        outstanding: dict[Future, tuple[K, Prompt, int]] = {}
        exhausted = False
        try:
            while True:
                # 2. Top up to `in_flight`, checking each prompt fits before it
                # is sent. One that does not is handed back at once and never
                # counts as outstanding, so the next one is pulled in its place.
                while not exhausted and len(outstanding) < self.in_flight:
                    item = next(work, None)
                    if item is None:
                        exhausted = True
                        break
                    key, prompt = item
                    if not self.model.fits(prompt):
                        log.warning("prompt %r does not fit the model's context; not sent", key)
                        meter.too_long += 1
                        self._report(meter.unit_done(), on_progress)
                        yield key, Outcome(
                            generation=Generation(text="", latency_ms=0, stop_reason="too_long"),
                            ok=False,
                            attempts=0,
                            failed=[TOO_LONG],
                        )
                        continue
                    outstanding[self.model.submit(prompt)] = (key, prompt, 0) # 0 means first attempt, not yet tried

                # 3. Done when the work is used up and nothing is outstanding.
                # we have in_flight number of outstandings
                if not outstanding:
                    return

                # 4. Settle the oldest finished prompt: retry it, or hand it back.
                done, _ = wait(outstanding, return_when=FIRST_COMPLETED)
                future = next(f for f in outstanding if f in done)
                key, prompt, n = outstanding.pop(future)
                n += 1
                try:
                    generation = future.result()
                except (ModelUnavailable, CancelledError):
                    raise
                except Exception as exc:
                    # The request failed; the model says it is still
                    # fit, or it would have raised ModelUnavailable. Re-send
                    # it as it was: there is no answer, so no rule to cite.
                    if n < self.max_attempts:
                        log.info(
                            "prompt %r attempt %d/%d failed with %r; retrying",
                            key, n, self.max_attempts, exc,
                        )
                        outstanding[self.model.submit(prompt)] = (key, prompt, n)
                        continue
                    log.warning("prompt %r: giving up after %d attempts; last error %r", key, n, exc)
                    self._report(meter.unit_done(), on_progress)
                    yield key, Outcome(
                        generation=Generation(text="", latency_ms=0, stop_reason="error"),
                        ok=False,
                        attempts=n,
                        failed=[ERROR],
                    )
                    continue
                failed = [c for c in checks if not c.passes(prompt, generation)]
                names = [c.name for c in failed]
                meter.attempt(Attempt(generation=generation, failed=names))

                # A retry is not checked for fit again: the repair margin is
                # already inside the model's slot, so the original's check covers it.
                if failed and n < self.max_attempts:
                    log.info(
                        "prompt %r attempt %d/%d failed %s; retrying",
                        key, n, self.max_attempts, names,
                    )
                    outstanding[self.model.submit(self._repair(prompt, failed))] = (key, prompt, n)
                    continue
                if failed:
                    log.warning(
                        "prompt %r: giving up after %d attempts; last failed %s", key, n, names
                    )
                self._report(meter.unit_done(), on_progress)
                yield key, Outcome(
                    generation=generation,
                    ok=not failed,
                    attempts=n,
                    failed=names,
                    parsed=parse(generation.text) if parse is not None and not failed else None,
                )
        finally:
            for future in outstanding:
                future.cancel()

    def handle(
        self,
        prompts: Sequence[Prompt],
        checks: Iterable[Check] = (),
        on_progress: ProgressFn | None = None,
    ) -> list[Outcome]:
        """Run every prompt to an outcome; outcomes come back in input order.

        `stream` over a list already in memory, for tests, notebooks and small
        sets. Blocking. There is no async twin: how many run at once is the
        model's decision, so an async version would add no parallelism. An
        async caller runs this on a worker thread --
        `await asyncio.to_thread(controller.handle, prompts, checks)` --
        with progress then reported from that thread.
        """
        prompts = list(prompts)
        outcomes: list[Outcome | None] = [None] * len(prompts)
        for i, outcome in self.stream(
            enumerate(prompts), checks, total=len(prompts), on_progress=on_progress
        ):
            outcomes[i] = outcome
        return outcomes  # type: ignore[return-value]

    # --- internals ----------------------------------------------------------

    def _repair(self, prompt: Prompt, failed: list[Check]) -> Prompt:
        """Append each failed check's message to the user turn."""
        rules = "\n".join(f"- {c.message}" for c in failed)
        note = (
            f"\n\nYour previous answer broke these rules:\n{rules}\n"
            "Answer again, following the instructions exactly."
        )
        return replace(prompt, user=prompt.user + note)

    def _ready(self, n_prompts: int | None) -> None:
        """model.ready(), translating ModelUnavailable into a readable failure."""
        try:
            self.model.ready()
        except ModelUnavailable as exc:
            which = "the" if n_prompts is None else f"the {n_prompts}"
            raise ModelUnavailable(
                f"The model could not be readied, so none of {which} "
                f"prompts were run.\n\n{exc}"
            ) from exc

    @staticmethod
    def _report(progress: Progress, on_progress: ProgressFn | None) -> None:
        of = "" if progress.total is None else f"/{progress.total}"
        log.info("%d%s %s", progress.done, of, progress.detail)
        if on_progress:
            on_progress(progress)
