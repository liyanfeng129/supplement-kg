"""Drives one `QwenModel`: readies it, runs prompts to outcomes, reports progress.

**Per model, not per workflow.** This controller knows exactly which model it
drives -- it imports `QwenModel` concretely, the one-controller-per-model rule
made literal -- and nothing about what the prompts are for. A future
`claude_controller` pairs with `claude_model` the same way.

**The model measures the work; the controller drives it.** The controller
hands the model the whole set and gets back a `Plan`: the prompts it can run,
in order, and the ones it cannot run at all. It then submits every runnable
prompt and handles each answer the moment it finishes. How many run at once is
the model's business -- the controller never learns a slot count, and never
waits on a batch: a slot that frees is refilled at once, which measured 1.38x
over one at a time where batch-by-batch measured 1.01x.

**Validation with retry lives here**, because the loop does. The checks come
with the prompt (see `repositories/prompts/`); the controller runs them
against each generation as it arrives and re-submits a failure at once --
repaired with each failed check's message -- until it passes or the budget is
spent. It can do that without understanding any check, because a check carries
its own message to the model. The prompt asks for a format and the controller
verifies it -- a prompt asking nicely is not a guarantee.

**Progress is counted here, measured below.** The model reports tokens per
generation; the controller turns completed units and the wall clock into
done-of-total, throughput and an ETA. Wall clock rather than summed latency,
because concurrent generations overlap: four at 6s each is 6s, not 24.

A check is a `Check` (see `models/base_model.py`): a name, a `passes(prompt,
generation)` test, and a message. It is run against the *original* prompt, so
a rule about the input sees the input, not a repaired copy. A check that raises
is a bug in the check and propagates.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from engine import config
from engine.models.base_model import (
    Check,
    Generation,
    ModelUnavailable,
    Plan,
    Progress,
    ProgressFn,
    Prompt,
)
from engine.models.qwen_model import QwenModel

log = logging.getLogger(__name__)

# Used when `configs/qwen.toml` has no `[controller].max_attempts`.
_FALLBACK_MAX_ATTEMPTS = 3

# The failure name given to a prompt the model's plan says cannot run.
TOO_LONG = "fits_context"


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
    Outcomes come back index-aligned with the prompts, so a service maps them
    to its records by position.
    """

    generation: Generation  # the last one, passing or not
    ok: bool
    attempts: int
    failed: list[str]  # names from the final attempt


def load_max_attempts(path: Path | None = None) -> int:
    """`[controller].max_attempts` from the TOML, or the fallback.

    A value that is not an integer falls back like a missing one: a broken
    config file must not raise at construction.
    """
    table = config.read_toml(path or config.QWEN_CONFIG_PATH).get("controller")
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
    the plan is made -- so a 40s load never inflates the ETA.
    """

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.tokens = 0
        self.completion_tokens = 0
        self.started = _now()

    def attempt(self, attempt: Attempt) -> None:
        usage = attempt.generation.usage
        self.tokens += usage.total_tokens
        self.completion_tokens += usage.completion_tokens

    def unit_done(self) -> Progress:
        self.done += 1
        elapsed = _now() - self.started
        parts = [f"{self.tokens} tok"]
        if elapsed > 0:
            parts.append(f"{self.completion_tokens / elapsed:.1f} tok/s")
        remaining = self.total - self.done
        if remaining:
            parts.append(f"eta {_duration(elapsed / self.done * remaining)}")
        return Progress(done=self.done, total=self.total, detail=" · ".join(parts))


class _Run:
    """One run's bookkeeping: checks, budget, repair, outcomes, progress.

    Kept apart from `handle`'s loop so the rules read without the waiting:
    `handle` submits what `start()` and `settle()` ask for, and nothing here
    knows how answers arrive.
    """

    def __init__(self, controller: "QwenController", prompts, checks, plan: Plan, on_progress):
        self.controller = controller
        self.prompts = prompts
        self.checks = checks
        self.plan = plan
        self.on_progress = on_progress
        self.outcomes: list[Outcome | None] = [None] * len(prompts)
        self.attempts = [0] * len(prompts)
        self.meter = _Meter(len(prompts))

    def start(self) -> list[tuple[int, Prompt]]:
        """Report what cannot run; return what to submit first, in plan order."""
        for i in self.plan.too_long:
            log.warning("prompt %d does not fit the model's context; not sent", i)
            self._finish(
                i,
                Outcome(
                    generation=Generation(text="", latency_ms=0, stop_reason="too_long"),
                    ok=False,
                    attempts=0,
                    failed=[TOO_LONG],
                ),
            )
        return [(i, self.prompts[i]) for i in self.plan.order]

    def settle(self, i: int, generation: Generation) -> Prompt | None:
        """Check one answer: finish the unit, or return the repair to re-submit."""
        self.attempts[i] += 1
        n, budget = self.attempts[i], self.controller.max_attempts
        failed = [c for c in self.checks if not c.passes(self.prompts[i], generation)]
        names = [c.name for c in failed]
        self.meter.attempt(Attempt(generation=generation, failed=names))

        if failed and n < budget:
            log.info("prompt %d attempt %d/%d failed %s; retrying", i, n, budget, names)
            return self.controller._repair(self.prompts[i], failed)
        if failed:
            log.warning("prompt %d: giving up after %d attempts; last failed %s", i, n, names)
        self._finish(i, Outcome(generation=generation, ok=not failed, attempts=n, failed=names))
        return None

    def result(self) -> list[Outcome]:
        missing = [i for i, o in enumerate(self.outcomes) if o is None]
        if missing:
            raise RuntimeError(f"the model's plan left prompts {missing} unassigned")
        return self.outcomes  # type: ignore[return-value]

    def _finish(self, i: int, outcome: Outcome) -> None:
        self.outcomes[i] = outcome
        self.controller._report(self.meter.unit_done(), self.on_progress)


class QwenController:
    """Holds one model and runs prompts through it to outcomes."""

    def __init__(self, model: QwenModel, max_attempts: int | None = None):
        """Holds one model. Budget from configs/qwen.toml unless overridden."""
        self.model = model
        self.max_attempts = (
            load_max_attempts() if max_attempts is None else max_attempts
        )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")

    # --- surface ------------------------------------------------------------

    def handle(
        self,
        prompts: Sequence[Prompt],
        checks: Iterable[Check] = (),
        on_progress: ProgressFn | None = None,
    ) -> list[Outcome]:
        """Run every prompt to an outcome, as many at once as the model allows.

        Blocking. There is no async twin: how many run at once is the model's
        decision, so an async version would add no parallelism. An async
        caller runs this on a worker thread --
        `await asyncio.to_thread(controller.handle, prompts, checks)` --
        with progress then reported from that thread.
        """
        prompts, checks = list(prompts), list(checks)
        if not prompts:
            return []

        self._ready(len(prompts))
        run = _Run(self, prompts, checks, self.model.plan(prompts), on_progress)
        pending: dict[Future, int] = {}
        try:
            for i, prompt in run.start():
                pending[self.model.submit(prompt)] = i
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i = pending.pop(future)
                    retry = run.settle(i, future.result())
                    if retry is not None:
                        pending[self.model.submit(retry)] = i
        finally:
            for future in pending:
                future.cancel()
        return run.result()

    # --- internals ----------------------------------------------------------

    def _repair(self, prompt: Prompt, failed: list[Check]) -> Prompt:
        """Append each failed check's message to the user turn."""
        rules = "\n".join(f"- {c.message}" for c in failed)
        note = (
            f"\n\nYour previous answer broke these rules:\n{rules}\n"
            "Answer again, following the instructions exactly."
        )
        return replace(prompt, user=prompt.user + note)

    def _ready(self, n_prompts: int) -> None:
        """model.ready(), translating ModelUnavailable into a readable failure."""
        try:
            self.model.ready()
        except ModelUnavailable as exc:
            raise ModelUnavailable(
                f"The model could not be readied, so none of the {n_prompts} "
                f"prompts were run.\n\n{exc}"
            ) from exc

    @staticmethod
    def _report(progress: Progress, on_progress: ProgressFn | None) -> None:
        log.info("%d/%d %s", progress.done, progress.total, progress.detail)
        if on_progress:
            on_progress(progress)
