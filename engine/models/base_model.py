"""What every model is, and the shapes that cross its boundary.

A model owns **its own lifecycle and its own hosting strategy**: it loads
itself, checks its own memory, enforces one instance when it holds something
exclusive (a GPU), and closes itself. It
does not know what workflow is running it -- there is no "passage" and no
"purpose/risk" at this layer, only a prompt in and a generation out. Moving
`QwenModel` from in-process llama.cpp to `llama-server` therefore changes
nothing above it.

**The shapes live here, not in a separate `schema.py`.** `Prompt`,
`Generation`, `Embedding`, `Usage` and `Progress` are the model boundary, and the boundary
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

import numpy as np


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
    runtime's fast path to report something nobody reads. `total` is None
    when the run streams work of unknown size; there is then no ETA.
    """

    done: int
    total: int | None
    detail: str = ""


@dataclass(frozen=True)
class Generation:
    """One model answer, with what it cost.

    `text` is raw: parsing it into claims is workflow logic and belongs in a
    service. The model does not know whether this text is JSON, lines, or
    nothing useful at all.

    `stop_reason` is carried because a generation cut off at `max_tokens` and
    one that finished are indistinguishable from the text alone, and the
    difference is the single most useful signal when output looks truncated.

    `model` says what answered, for a model that cannot pin its own runtime: a
    CLI model records `<model id>@<cli>-<version>` here, because the CLI updates
    itself. Empty for a model whose settings already say what ran.
    """

    text: str
    latency_ms: int
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    model: str = ""


@dataclass(frozen=True)
class Embedding:
    """The vectors for one request's texts, with what they cost.

    Row `i` is text `i`'s vector: float32, shape `(n, dim)`, each row L2-
    normalised, so cosine similarity is a dot product. `usage.prompt_tokens`
    is the tokens embedded; `completion_tokens` is 0, as nothing is generated.
    `Usage` is reused so a token means the same thing for both kinds of model.
    """

    vectors: np.ndarray
    latency_ms: int
    usage: Usage


@dataclass(frozen=True)
class Check:
    """One rule an answer must follow, and what to tell the model when it does not.

    Atomic: one rule per check, so a failed answer is told exactly which rules
    it broke and nothing else. `passes` gets the prompt as well as the
    generation because some rules are about the input -- "every claim is copied
    from the label" needs the label.

    `message` is written *to the model*: on a failure it is appended to the
    retried prompt as an instruction. `name` is written for people -- logs and
    `Outcome.failed` -- and the model never sees it.
    """

    name: str
    passes: Callable[[Prompt, Generation], bool]
    message: str


ProgressFn = Callable[[Progress], None]


class ModelUnavailable(RuntimeError):
    """The model cannot run, so the caller must stop rather than degrade.

    Raised instead of falling back to a slower or different path. The failure
    this guards against on the local model -- a silent CPU run -- produces
    correct rows at ~100x the cost, so nothing but the clock reveals it.
    """


class BaseModel(ABC):
    """The interface every model implements.

    Three verbs, plus three for work sets. `ready()` makes the model usable
    and is idempotent, so a caller may assert readiness without tracking
    whether it already did. `generate()` answers one prompt. `close()`
    releases whatever `ready()` took. `fits()` says whether one prompt can run
    at all; `cache_shared_prefix()` prepares the model for prompts like a
    sample of them; `submit()` queues one prompt and returns a future.

    A caller keeps more prompts submitted than the model runs at once and
    takes answers as they finish, so a model that runs several prompts at once
    keeps every slot busy -- a slot frees, the next queued prompt takes it.
    The defaults run one prompt at a time and accept every prompt, so a model
    with no concurrency implements only `generate()`.

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

    def fits(self, prompt: Prompt) -> bool:
        """Can this prompt run at all? The default has no limit to check."""
        return True

    def cache_shared_prefix(self, sample: Sequence[Prompt]) -> None:
        """Prepare for prompts like `sample`. The default has nothing to cache."""

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


class BaseEmbedModel(ABC):
    """The interface every embedding model implements.

    A separate class beside `BaseModel`, not a subclass: an embedding model
    has no prompt and no generation, only texts in and vectors out. The verbs
    mirror `BaseModel`'s, with the same defaults: one request at a time, and
    every text accepted.

    **The instruction is the caller's wording; the template is the model's.**
    How an instruction is combined with a text is specific to each embedding
    model, as ChatML is to Qwen, so the model applies it. `instruction=None`
    embeds the raw text.
    """

    def ready(self) -> None:
        """Load and verify. Idempotent; safe to call before every batch."""

    @abstractmethod
    def embed(self, texts: Sequence[str], instruction: str | None = None) -> Embedding:
        """One vector per text, rows in input order."""

    def fits(self, text: str, instruction: str | None = None) -> bool:
        """Can this text be embedded at all? The default has no limit to check."""
        return True

    def submit(self, texts: Sequence[str], instruction: str | None = None) -> Future[Embedding]:
        """Queue one request; the future resolves to its embedding.

        The default embeds before returning. A concurrent model overrides
        this and bounds how many run at once itself.
        """
        future: Future[Embedding] = Future()
        try:
            future.set_result(self.embed(texts, instruction))
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
