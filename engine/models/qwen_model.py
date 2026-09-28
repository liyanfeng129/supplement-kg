"""Qwen on llama.cpp's server: starts it, guards the GPU, answers prompts four at once.

**Hosted by `llama-server`, and that is this file's private business.** The
model starts the server itself in `ready()`, talks to it over HTTP on
localhost, and stops it in `close()`; nothing above knows a server exists. The
runtime was chosen by measurement (`benchmarks/parallel_slots.py`, run
`parallel-20260923T145320Z`): on 32 labels the server with one slot took 74.6s
against 99.5s for in-process `llama_cpp`, and four slots took 53.7s -- 1.85x --
with grounding at 1.0 throughout. Six and eight slots were no faster than four:
this work is mostly prompt reading, which one request already runs at full
width, so more slots only queue.

**Slots are this model's concurrency, and it keeps them full.** A controller
hands over its whole work set; `plan()` tokenises every prompt and sets aside
any that cannot fit a slot. The controller then `submit()`s every prompt and
takes answers as they finish; a pool of `n_parallel` workers means a slot that
frees takes the next queued prompt at once. Batches with a barrier were
measured and rejected: waiting for each batch's slowest member left slots idle
and ran no faster than one at a time (73.5s vs 73.9s, against 53.7s for a
continuous pool; run parallel-20260923T152816Z).

**Tuned by measurement, not by guess.** Swept on 64 labels at 4 slots, load
excluded (runs parallel-20260923T154114Z and -155732Z): defaults 101-105s;
the model's own MTP draft head for speculative decoding 77s; that plus
2,048-token ubatches plus primed slots 65.5s -- 1.57x over the server's
defaults, ~3x over the in-process path this replaced. Five drafted tokens
were 2.5% faster than three -- noise -- for 1.2 GB more VRAM, so three.
`--kv-unified` and a 1,024 ubatch changed nothing. Greedy output is the
same model's either way: speculation is verified by the full model.

**GPU-only, asserted around the load.** Two failures both end in a silent CPU
run -- the CUDA backend failing to initialise, and the card being occupied so
the weights do not fit -- and neither is visible in the output: the rows are
fine, only the clock differs. So free VRAM is checked before starting, and the
load must then take most of the weights' size in VRAM; a failure names whatever
process is holding the card.

**Ready means it talks sense.** This architecture's CUDA path can load cleanly
and emit fluent garbage, and the server links a different ggml build from the
one in the venv. `ready()` ends by asking for a greeting and refuses a reply
that is not language -- a weak check, stated as such on `incoherence()`.

**Constrained output is a schema, never a format name.** `Prompt.schema`
present is sent as the server's `json_schema`, which it compiles to a grammar
that constrains every token; absent, the sampler runs free. The grammar costs
~4x in decode speed, and that is the caller's decision, made by passing a
schema or not.

**Settings come from `configs/qwen.toml`.** Hyperparameters are experiment
inputs, so they live in git where a run's manifest can quote them. An explicit
constructor argument still wins, and a missing or malformed file falls back to
`_FALLBACKS` -- constructing a model is free and must never raise.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from engine.models.base_model import (
    BaseModel,
    Generation,
    ModelUnavailable,
    Plan,
    Prompt,
    Usage,
)

# Four slots at 5,120 tokens with the MTP draft head and 2,048-token ubatches
# peaked at ~20.9 GB; the margin is for whatever the driver keeps.
MIN_VRAM_MIB = 21_500

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPO_ROOT / "configs" / "qwen.toml"

# What `load_defaults` returns when the file says nothing. Mirrors the committed
# file except for the paths: a machine-specific path has no sane fallback, so
# an empty one makes `ready()` refuse rather than guess.
_FALLBACKS: dict[str, Any] = {
    "model_path": "",
    "server_path": "",
    "n_parallel": 4,
    "slot_ctx": 5120,
    "ubatch": 2048,
    "speculative": "draft-mtp",
    "draft_max": 3,
    "min_vram_mib": MIN_VRAM_MIB,
    "seed": 0,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_tokens": 2000,
    "enable_thinking": False,
}

# The tables whose keys are constructor arguments. `[controller]` is read by
# the controller and deliberately not flattened in here.
_MODEL_TABLES = ("model", "sampling")

# Tokens a prompt must leave free beyond its own output budget: a controller's
# repair note ("your previous answer failed these checks: ...") lengthens a
# re-sent prompt, and a retry must not be the thing that overflows the slot.
REPAIR_MARGIN = 64

# A load that takes less VRAM than this share of the weights file did not put
# the weights on the GPU. Measured loads take more than the file (KV cache and
# buffers on top), so the threshold only has to rule out a CPU fallback.
_OFFLOAD_SHARE = 0.9

_HEALTH_TIMEOUT_S = 900
_REQUEST_TIMEOUT_S = 1800


class ServerError(RuntimeError):
    """The server answered with an error; the message is the server's own."""


def read_config(path: Path = DEFAULTS_PATH) -> dict:
    """The parsed TOML, or {} when the file is missing or malformed."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def load_defaults(path: Path = DEFAULTS_PATH) -> dict:
    """Flatten the TOML into constructor kwargs. Missing file -> hardcoded defaults.

    Only keys `QwenModel` accepts are taken, so a typo in the file is ignored
    rather than turned into a `TypeError` at construction.
    """
    config = read_config(path)
    defaults = dict(_FALLBACKS)
    for table in _MODEL_TABLES:
        section = config.get(table)
        if isinstance(section, dict):
            defaults.update({k: v for k, v in section.items() if k in _FALLBACKS})
    return defaults


# The hello check's request. Asked for English because the coherence test
# looks for Latin-script words; capped short because it runs on every load.
HELLO_PROMPT = Prompt(
    system="You are a helpful assistant. Reply with one short sentence in English.",
    user="Hello",
    max_tokens=32,
)

# Longest reply to "Hello" still taken as a greeting. 32 tokens is ~150 chars;
# the slack is for tokenisers that split oddly, not for rambling.
_HELLO_MAX_CHARS = 400

_WORD = re.compile(r"[A-Za-z]{2,}")


def incoherence(text: str) -> str | None:
    """Why `text` is not a coherent reply to a greeting, or None if it is.

    Deliberately weak: non-empty, plausible length, made mostly of words. It
    catches a runtime emitting garbage -- stray byte tokens, symbol soup,
    runaway output. It will not catch a model that is subtly degraded but
    still says hello fine; a stronger check needs domain fixtures, and those
    belong above this layer.
    """
    stripped = text.strip()
    if not stripped:
        return "the reply was empty"
    if len(stripped) > _HELLO_MAX_CHARS:
        return f"the reply was too long for a greeting ({len(stripped)} chars)"

    words = _WORD.findall(stripped)
    letters = sum(len(w) for w in words)
    visible = sum(1 for c in stripped if not c.isspace())
    if not words or letters / visible < 0.5:
        return "the reply had no words in it, or too few to be language"
    return None


def gpu_holders() -> list[dict]:
    """Processes currently holding GPU memory, via nvidia-smi.

    Used only to explain a failure. The overwhelmingly common cause of "the GPU
    is unavailable" here is a previous run that was never killed and still has
    the weights resident, and naming that process is the difference between a
    one-command fix and a debugging session.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    holders = []
    for line in out.stdout.strip().splitlines():
        parts = [f.strip() for f in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            holders.append({"pid": parts[0], "name": parts[1], "used_mib": parts[2]})
    return holders


def free_vram_mib() -> int | None:
    """Free VRAM on device 0, or None when it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()[:1]
    return int(first[0].strip()) if first and first[0].strip().isdigit() else None


def _unavailable_message(reason: str) -> str:
    """Explain why the GPU is unusable, and name whatever is holding it."""
    lines = [
        f"Refusing to run: {reason}",
        "",
        "This model is GPU-only by design. Falling back to CPU is not an",
        "option here -- it is ~100x slower, so a corpus run would take years",
        "rather than days, and llama.cpp does it silently: no error, correct",
        "rows, a perfect grounding rate, and only the clock to tell you.",
    ]

    holders = gpu_holders()
    if holders:
        lines += ["", "The GPU is currently held by:"]
        lines += [
            f"  PID {h['pid']}  {h['name']}  ({h['used_mib']} MiB)" for h in holders
        ]
        lines += [
            "",
            "This is usually a previous run that was never killed -- a notebook",
            "kernel with the model loaded counts. Stop it and retry:",
            f"  kill {' '.join(h['pid'] for h in holders)}",
        ]
    else:
        lines += [
            "",
            "No process is holding the GPU, so this is not a leftover run.",
            "Check that GGML_CUDA_NO_PINNED=1 reaches the server (ggml's CUDA",
            "init fails on pinned-memory allocation on this machine), and that",
            "the server's own libggml-cuda.so loads.",
        ]
    return "\n".join(lines)


def shared_prefix(rendered: Sequence[str]) -> str:
    """The text every rendered prompt starts with, cut after the last full turn.

    Cut at a turn boundary (`<|im_end|>\n`) so the prefix tokenises exactly as
    it does inside each full prompt; a cut mid-text could merge differently
    with what follows. Fewer than two prompts share nothing worth priming.
    """
    if len(rendered) < 2:
        return ""
    common = os.path.commonprefix(list(rendered))
    end = common.rfind("<|im_end|>\n")
    return common[: end + len("<|im_end|>\n")] if end >= 0 else ""


def _die_with_parent() -> None:
    """Runs in the server's child process: SIGTERM it when this process dies.

    A server that outlives its Python process keeps 18 GB of weights on the
    card with nothing left to stop it -- the "previous run never killed"
    failure, created by us. Linux-only, which this machine is.
    """
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


class QwenModel(BaseModel):
    """One server, `n_parallel` slots, many generations.

    The server starts on `ready()` and is reused: loading 17 GB per call would
    dominate any run. Construction is free, so a caller may build one to read
    its settings without touching the GPU.
    """

    def __init__(
        self,
        model_path: str | None = None,
        server_path: str | None = None,
        n_parallel: int | None = None,
        slot_ctx: int | None = None,
        ubatch: int | None = None,
        speculative: str | None = None,
        draft_max: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        min_vram_mib: int | None = None,
        verbose: bool = False,
    ):
        """Every setting left as None comes from `configs/qwen.toml`.

        `None` rather than a literal default so the file can be the one place
        a setting is stated; `0` and `False` are real values and still win.
        `slot_ctx` is per slot: the server is started with `n_parallel` times
        it, split evenly, so every slot can hold any prompt `plan()` accepts.
        """
        defaults = load_defaults()

        def pick(value, key):
            return defaults[key] if value is None else value

        self.model_path = pick(model_path, "model_path")
        self.server_path = pick(server_path, "server_path")
        self.n_parallel = pick(n_parallel, "n_parallel")
        self.slot_ctx = pick(slot_ctx, "slot_ctx")
        self.ubatch = pick(ubatch, "ubatch")
        self.speculative = pick(speculative, "speculative")
        self.draft_max = pick(draft_max, "draft_max")
        self.seed = pick(seed, "seed")
        self.temperature = pick(temperature, "temperature")
        self.top_p = pick(top_p, "top_p")
        self.top_k = pick(top_k, "top_k")
        self.max_tokens = pick(max_tokens, "max_tokens")
        self.enable_thinking = pick(enable_thinking, "enable_thinking")
        self.min_vram_mib = pick(min_vram_mib, "min_vram_mib")
        self.verbose = verbose

        self._proc: Any = None
        self._pool: ThreadPoolExecutor | None = None
        self._primed = ""  # the shared prefix every slot currently holds
        self._base_url = ""
        self.log_path: Path | None = None

    # --- lifecycle ----------------------------------------------------------

    def ready(self) -> None:
        """Check the GPU, start the server, check the model talks sense. Idempotent.

        Ready means *loaded and generating language*, not merely loaded: a
        model that reports ready without saying hello could be lying. The
        guard below makes the check run once per start, not once per batch.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = None

        if not self.model_path:
            raise ModelUnavailable(
                "Refusing to run: no model_path is set. Pass one, or set "
                f"[model].model_path in {DEFAULTS_PATH}."
            )
        if not self.server_path or not Path(self.server_path).is_file():
            raise ModelUnavailable(
                f"Refusing to run: no llama-server at {self.server_path!r}. Pass "
                f"server_path, or set [model].server_path in {DEFAULTS_PATH}."
            )

        free_before = self._check_vram()
        self._start()
        try:
            self._wait_healthy()
            self._check_offload(free_before)
            self._check_alive()
        except BaseException:
            self.close()
            raise
        # One worker per slot: the server would queue any excess anyway, and
        # a bounded pool keeps the waiting line in this process, where
        # `close()` can cancel it.
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, self.n_parallel), thread_name_prefix="qwen-slot"
        )

    def close(self) -> None:
        """Stop the server and free the card. Idempotent.

        Queued prompts not yet sent are cancelled; their futures raise
        `CancelledError`.
        """
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        self._primed = ""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _check_vram(self) -> int | None:
        """Refuse to start unless the card has room; return the free MiB seen.

        Checked on every start rather than once at import, because the card
        can be taken between one run and the next, and the symptom of missing
        it is only the clock.
        """
        free = free_vram_mib()
        if self.min_vram_mib and free is not None and free < self.min_vram_mib:
            raise ModelUnavailable(
                _unavailable_message(
                    f"only {free} MiB of VRAM free, and the model needs about "
                    f"{self.min_vram_mib} MiB."
                )
            )
        return free

    def _server_args(self, port: int) -> list[str]:
        return [
            str(self.server_path),
            "-m", str(self.model_path),
            "-ngl", "999",
            "-np", str(self.n_parallel),
            "-c", str(self.n_parallel * self.slot_ctx),
            # Each slot gets its own `slot_ctx`, so `plan()`'s fit check is
            # exact rather than a hope about a shared pool.
            "--no-kv-unified",
            # Prompt reading is most of this work; a 2,048-token physical
            # batch reads a label in one pass. Only paired with priming: alone
            # it moved the cache checkpoints and re-read 1,460 tokens a label.
            "-ub", str(self.ubatch),
            "-b", str(max(self.ubatch, 2048)),
            *(
                ["--spec-type", self.speculative, "--spec-draft-n-max", str(self.draft_max)]
                if self.speculative and self.speculative != "none"
                else []
            ),
            "-s", str(self.seed),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--no-webui",
            *(["--verbose"] if self.verbose else []),
        ]

    def _start(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._base_url = f"http://127.0.0.1:{port}"

        fd, name = tempfile.mkstemp(prefix="qwen-server-", suffix=".log")
        self.log_path = Path(name)
        env = {**os.environ, "GGML_CUDA_NO_PINNED": "1"}
        with os.fdopen(fd, "w") as log:
            self._proc = self._spawn(self._server_args(port), log, env)

    def _spawn(self, args: list[str], log, env: dict) -> Any:
        """Start the server process. Separate so a test can stand in for it."""
        return subprocess.Popen(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=_die_with_parent,
        )

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + _HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ModelUnavailable(
                    _unavailable_message(
                        f"llama-server exited with code {self._proc.returncode} "
                        f"while loading.\n\nLast lines of {self.log_path}:\n"
                        + self._log_tail()
                    )
                )
            try:
                if self._request("/health", timeout=2).get("status") == "ok":
                    return
            except (OSError, ValueError, ServerError):
                pass  # not listening yet, or 503 "Loading model"
            time.sleep(1)
        raise ModelUnavailable(
            f"Refusing to run: llama-server did not become healthy within "
            f"{_HEALTH_TIMEOUT_S}s. See {self.log_path}."
        )

    def _check_offload(self, free_before: int | None) -> None:
        """Refuse a load that did not put the weights on the GPU.

        The server does not log offload at its default verbosity, so this
        measures instead: the load must take most of the weights file's size
        in VRAM. A CPU fallback takes a few hundred MiB.
        """
        free_after = free_vram_mib()
        try:
            weights_mib = os.path.getsize(self.model_path) / 2**20
        except OSError:
            return
        if free_before is None or free_after is None:
            return
        taken = free_before - free_after
        if taken < _OFFLOAD_SHARE * weights_mib:
            raise ModelUnavailable(
                _unavailable_message(
                    f"the load took only {taken} MiB of VRAM for "
                    f"{weights_mib:.0f} MiB of weights, so they are not on the GPU."
                )
            )

    def _check_alive(self) -> None:
        """Generate a greeting; raise ModelUnavailable if it isn't coherent language."""
        reply = self.generate(HELLO_PROMPT).text
        why = incoherence(reply)
        if why is None:
            return
        raise ModelUnavailable(
            "Refusing to run: the model loaded but its reply to \"Hello\" was "
            f"not coherent language -- {why}.\n\n"
            f"It said: {reply[:200]!r}\n\n"
            "This is the known failure of this model's CUDA path: a clean load\n"
            "followed by fluent garbage. The usual cause is a stale\n"
            "libggml-cuda.so surviving a rebuild -- check which one the\n"
            f"server at {self.server_path} actually loads (ldd)."
        )

    def _log_tail(self, n: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except (OSError, AttributeError):
            return "(no log)"

    # --- generation ---------------------------------------------------------

    def generate(self, prompt: Prompt) -> Generation:
        """Answer one prompt.

        Safe to call from several threads at once -- the server queues
        requests into its slots -- which is how `submit()`'s workers run.
        No inline retry: validation and retry are the controller's.
        """
        self.ready()
        body: dict[str, Any] = {
            "prompt": self._render(prompt),
            "n_predict": prompt.max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "stop": list(prompt.stop) or ["<|im_end|>"],
            # Each slot keeps its last prompt, so the shared system prompt is
            # read once per slot rather than once per request.
            "cache_prompt": True,
        }
        if prompt.schema is not None:
            body["json_schema"] = prompt.schema

        started = time.monotonic()
        out = self._request("/completion", body)
        latency_ms = int((time.monotonic() - started) * 1000)
        return Generation(
            text=out.get("content") or "",
            latency_ms=latency_ms,
            usage=Usage(
                prompt_tokens=int(out.get("tokens_evaluated") or 0),
                completion_tokens=int(out.get("tokens_predicted") or 0),
            ),
            stop_reason=str(out.get("stop_type") or ""),
        )

    def submit(self, prompt: Prompt) -> Future[Generation]:
        """Queue one prompt; at most `n_parallel` run at once, the rest wait."""
        self.ready()
        return self._pool.submit(self.generate, prompt)

    def plan(self, prompts: Sequence[Prompt]) -> Plan:
        """Measure a work set: what fits a slot, in the order given.

        Every prompt is measured in this model's own tokens -- only the model
        knows its tokenizer. A prompt whose tokens, output budget and repair
        margin exceed one slot goes to `too_long` instead of failing mid-run.

        The order is left as given. Longest-first was tried and measured worse:
        its first dozen requests re-read their whole prompt instead of reusing
        the cached system prompt (32,371 prompt tokens read vs 25,317).
        """
        self.ready()
        with ThreadPoolExecutor(max_workers=8) as pool:
            sizes = list(pool.map(self.count_tokens, prompts))

        fits, too_long = [], []
        for i, (prompt, size) in enumerate(zip(prompts, sizes)):
            need = size + (prompt.max_tokens or self.max_tokens) + REPAIR_MARGIN
            (fits if need <= self.slot_ctx else too_long).append(i)

        self._prime([prompts[i] for i in fits])
        return Plan(order=tuple(fits), too_long=tuple(too_long))

    def _prime(self, prompts: Sequence[Prompt]) -> None:
        """Have every slot read the prefix these prompts share, once.

        Every prompt of a run starts with the same system turn -- ~1,600 of
        ~2,000 tokens -- and a slot that already holds it reads only the rest.
        On this hybrid model a slot can only resume from a saved checkpoint,
        and a prefix read on its own leaves one exactly at its end; left to
        chance, the first requests re-read everything. Measured on 64 labels:
        prompt tokens read per label 1,460 -> 444 with `ubatch = 2048`.

        Sent to all slots at once so each gets one. Skipped when the slots
        already hold this prefix, so a second `plan()` for the same run costs
        nothing.
        """
        prefix = shared_prefix([self._render(p) for p in prompts])
        if not prefix or prefix == self._primed:
            return
        body = {"prompt": prefix, "n_predict": 1, "temperature": 0.0, "cache_prompt": True}
        n = max(1, self.n_parallel)
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda _: self._request("/completion", body), range(n)))
        self._primed = prefix

    def count_tokens(self, prompt: Prompt) -> int:
        """Tokens this prompt occupies in a slot, as rendered."""
        self.ready()
        out = self._request("/tokenize", {"content": self._render(prompt)})
        return len(out.get("tokens") or [])

    def _request(self, path: str, body: dict | None = None, timeout: float = _REQUEST_TIMEOUT_S) -> dict:
        """GET (no body) or POST JSON to the server; the parsed reply.

        A request the server rejects raises with the server's own message --
        "exceeds the available context", say -- rather than a bare HTTP code.
        """
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self._base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise ServerError(f"llama-server {path} -> {exc.code}: {detail}") from exc

    def _render(self, prompt: Prompt) -> str:
        """Lay the prompt out as ChatML, with thinking suppressed explicitly.

        Built here rather than via the server's chat endpoint so the template's
        `enable_thinking` branch cannot be skipped: it is the only thing
        standing between a run and `reasoning_effort: xhigh`, which this
        model's template applies by default -- so the trailing
        `<think>\\n\\n</think>` the template would emit is written here, where it
        is visible and testable rather than hoped for.

        ChatML is a fact about this model, which is why it lives at this layer:
        nothing above should know these tokens exist.
        """
        header = (
            f"<|im_start|>system\n{prompt.system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt.user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        opener = "<think>\n" if self.enable_thinking else "<think>\n\n</think>\n\n"
        return header + opener
