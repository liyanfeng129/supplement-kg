"""What every model is, and the shapes that cross its boundary.

A model owns **its own lifecycle and its own hosting strategy**: it loads
itself, checks its own memory, enforces one instance, and closes itself. It
does not know what workflow is running it -- there is no "passage" and no
"purpose/risk" at this layer, only a prompt in and a generation out. Moving
`QwenModel` from in-process llama.cpp to `llama-server` therefore changes
nothing above it.

**The shapes live here, not in a separate `schema.py`.** `Prompt`,
`Generation`, `Usage` and `Progress` are the model boundary, and the boundary
and the thing that defines it drift apart the moment they are edited in
different files for different reasons. Every model imports them from here.

**Why a shape at all, rather than a tuple.** The driver this replaces returned
`(text, latency_ms, usage)`: a caller learned what was in it by reading the
function body, and adding a field meant finding every unpack site. A named
shape states at the signature what comes back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True)
class Usage:
    """Token counts for one generation, as the runtime reported them.

    Both default to 0 rather than None: a runtime that does not report counts
    yields a row that still sums, and throughput stays computable. A missing
    count is visible as a zero against a non-zero latency.

    Without these, throughput can only be reported per *label*, which hides the
    thing that actually varies -- a long label and a short one differ by an
    order of magnitude in work while looking identical at one-per-label.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class Prompt:
    """One request to a model: what to say, and what shape to answer in.

    `system` and `user` are plain text. How they become the wire format --
    ChatML with special tokens, an Anthropic `messages` array -- is the
    model's business and nothing above it may assume either.

    **`schema` is how a caller asks for constrained output**, and it is a
    schema rather than a format name on purpose. A model that took `fmt="json"`
    would force every caller to know which formats that particular model
    supports; a schema is a fact about the answer wanted, and each model
    honours it however it can -- llama.cpp compiles a GBNF grammar, an API
    model passes it as structured output. `schema=None` means unconstrained,
    which is what the `lines` format wants.

    `stop` is a tuple because a frozen dataclass with a mutable default is a
    shared-state bug waiting to happen, and because stop sequences are a fixed
    property of the request, not a list something appends to later.
    """

    system: str
    user: str
    schema: dict | None = None
    stop: tuple[str, ...] = ()
    max_tokens: int | None = None


@dataclass(frozen=True)
class Progress:
    """How far a batch has got, as the model sees it.

    The model emits these; a controller relays them to a log. A frontend that
    wants to display progress subscribes to the controller, and one that does
    not gets a quiet library -- which is the whole reason this is a callback
    and not a `print` inside a loop.

    `done` and `total` count whole units of work, not tokens: at seconds per
    unit the useful granularity is the unit, and token streaming costs the
    runtime's fast path to report something nobody reads.
    """

    done: int
    total: int
    detail: str = ""

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


@dataclass(frozen=True)
class Generation:
    """One model answer, with what it cost.

    `text` is raw: parsing it into claims is workflow logic and belongs in a
    service. The model does not know whether this text is JSON, lines, or
    nothing useful at all.

    `stop_reason` is carried because a generation cut off at `max_tokens` and
    one that finished are indistinguishable from the text alone, and the
    difference is the single most useful signal when output looks truncated.
    """

    text: str
    latency_ms: int
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "latency_ms": self.latency_ms,
            "usage": self.usage.to_dict(),
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class Plan:
    """How a model will run a work set, decided by the model after measuring it.

    The controller hands a model the whole set; the model answers with this.
    `order` holds the indices it can run, in the order to submit them.
    `too_long` are indices it cannot run at all -- a prompt that does not fit
    its context -- stated up front rather than discovered as a failure halfway
    through a run.

    How many run at once is deliberately absent: that is the model's to
    enforce inside `submit()`, and a controller that knew it would be
    scheduling around one hosting strategy.
    """

    order: tuple[int, ...]
    too_long: tuple[int, ...] = ()


ProgressFn = Callable[[Progress], None]


class ModelUnavailable(RuntimeError):
    """The model cannot run, so the caller must stop rather than degrade.

    Raised instead of falling back to a slower or different path. The failure
    this guards against on the local model -- a silent CPU run -- produces
    correct rows at ~100x the cost, so nothing but the clock reveals it.
    """


class BaseModel(ABC):
    """The interface every model implements.

    Three verbs, plus two for work sets. `ready()` makes the model usable and
    is idempotent, so a caller may assert readiness without tracking whether
    it already did. `generate()` answers one prompt. `close()` releases
    whatever `ready()` took. `plan()` measures a work set and says what can
    run and in what order; `submit()` queues one prompt and returns a future.

    A caller submits a whole set and takes answers as they finish, so a model
    that runs several prompts at once keeps every slot busy -- a slot frees,
    the next queued prompt takes it. The defaults run one prompt at a time, so
    a model with no concurrency implements only `generate()`.

    Loading belongs in `ready()` rather than `__init__` so that constructing a
    model is free: a CLI can build one to read its configuration, and a test
    can construct one without a GPU. `generate()` calls `ready()` itself, so a
    caller that skips it is not punished with a confusing error -- the guard a
    caller can forget is not a guard.
    """

    def ready(self) -> None:
        """Load and verify. Idempotent; safe to call before every batch."""

    @abstractmethod
    def generate(self, prompt: Prompt) -> Generation:
        """Answer one prompt."""

    def plan(self, prompts: Sequence[Prompt]) -> Plan:
        """Measure a work set: what can run, and in what order to submit it."""
        return Plan(order=tuple(range(len(prompts))))

    def submit(self, prompt: Prompt) -> Future[Generation]:
        """Queue one prompt; the future resolves to its generation.

        The default answers before returning. A concurrent model overrides
        this and bounds how many run at once itself.
        """
        future: Future[Generation] = Future()
        try:
            future.set_result(self.generate(prompt))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def close(self) -> None:
        """Release resources. Idempotent."""

    def __enter__(self):
        self.ready()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
