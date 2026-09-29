"""Qwen on llama.cpp's server: one shared model, GPU-only, four prompts at once.

**One model per process.** `QwenModel()` always returns the same object, so
the 17 GB of weights are never loaded twice. Asking for different settings
restarts the server with them -- but only while nothing is running.

**Hosted by `llama-server`, and that is this file's private business.**
`ready()` starts the server, the model talks to it over HTTP on localhost,
and `close()` stops it. Nothing above knows a server exists.

**GPU-only.** A CPU fallback gives correct rows at ~100x the cost, and only
the clock shows it. So `ready()` refuses a card without room, a load that did
not reach the GPU, and a model that answers "Hello" in garbage.

**Settings come from `configs/qwen.toml`**; an explicit argument wins. Every
tuned number (4 slots, speculative decoding, 2,048-token batches, primed
slots, input order) was chosen by measurement: see
`docs/qwen-model-controller-design.md`, "Measurements behind the code".

Public: `ready`, `generate`, `fits`, `cache_shared_prefix`, `submit`, `close`. Everything else is private.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from engine import config
from engine.models.base_model import (
    BaseModel,
    Generation,
    ModelUnavailable,
    Prompt,
    Usage,
)

log = logging.getLogger(__name__)

# Used where `configs/qwen.toml` says nothing. The paths are empty on purpose:
# a machine-specific path has no sane fallback, so `ready()` refuses instead.
_FALLBACKS: dict[str, Any] = {
    "model_path": "",
    "server_path": "",
    "n_parallel": 4,
    "slot_ctx": 5120,
    "ubatch": 2048,
    "speculative": "draft-mtp",
    "draft_max": 3,
    "min_vram_mib": 21_500,  # measured peak ~20.9 GB, plus driver margin
    "seed": 0,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_tokens": 2000,
    "enable_thinking": False,
    "verbose": False,
}

# The TOML tables the model reads. `[controller]` is the controller's.
_MODEL_TABLES = ("model", "sampling")

# Room a prompt must leave in its slot for the controller's repair note, so a
# retry is never the thing that overflows it.
_REPAIR_MARGIN = 64

# A load taking less VRAM than this share of the weights file did not put the
# weights on the GPU (a CPU fallback takes a few hundred MiB).
_OFFLOAD_SHARE = 0.9

_HEALTH_TIMEOUT_S = 900
_REQUEST_TIMEOUT_S = 1800

# The hello check: English, because the coherence test looks for Latin-script
# words; short, because it runs on every start.
_HELLO_PROMPT = Prompt(
    system="You are a helpful assistant. Reply with one short sentence in English.",
    user="Hello",
    max_tokens=32,
)
_HELLO_MAX_CHARS = 400  # 32 tokens is ~150 chars; the slack is for odd tokenisers
_WORD = re.compile(r"[A-Za-z]{2,}")

# Where one ChatML turn ends. The shared prefix is cut here so it tokenises
# exactly as it does inside each full prompt.
_TURN_END = "<|im_end|>\n"


class ServerError(RuntimeError):
    """The server answered with an error; the message is the server's own."""


def _unavailable_message(reason: str) -> str:
    """Explain why the GPU is unusable, and name whatever process is holding it."""
    lines = [
        f"Refusing to run: {reason}",
        "",
        "This model is GPU-only by design. Falling back to CPU is not an",
        "option here -- it is ~100x slower, so a corpus run would take years",
        "rather than days, and llama.cpp does it silently: no error, correct",
        "rows, a perfect grounding rate, and only the clock to tell you.",
    ]

    # Who holds the card. Usually a previous run that was never killed.
    holders = []
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
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                parts = [f.strip() for f in line.split(",")]
                if len(parts) >= 3 and parts[0]:
                    holders.append(parts)
    except (OSError, subprocess.SubprocessError):
        pass

    if holders:
        lines += ["", "The GPU is currently held by:"]
        lines += [f"  PID {pid}  {name}  ({mib} MiB)" for pid, name, mib, *_ in holders]
        lines += [
            "",
            "This is usually a previous run that was never killed -- a notebook",
            "kernel with the model loaded counts. Stop it and retry:",
            f"  kill {' '.join(h[0] for h in holders)}",
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


class QwenModel(BaseModel):
    """The one Qwen model: one server, `n_parallel` slots, many generations.

    Build it with `QwenModel(...)` -- free, touches no GPU. `ready()` starts
    the server; `close()` stops it.
    """

    # The settings. All come from `configs/qwen.toml` unless passed.
    model_path: str
    server_path: str
    n_parallel: int  # prompts answered at once
    slot_ctx: int  # tokens per slot: prompt + output + repair margin must fit
    ubatch: int
    speculative: str  # "none" turns speculative decoding off
    draft_max: int
    min_vram_mib: int
    seed: int
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    enable_thinking: bool
    verbose: bool

    log_path: Path | None  # the server's log file, once started

    # Private state, set up in `__new__`.
    _proc: Any  # the llama-server process
    _pool: ThreadPoolExecutor | None  # one worker per slot, created by ready()
    _base_url: str
    _primed: str  # the shared prefix every slot currently holds
    _in_flight: int  # prompts queued or being answered
    _in_flight_lock: threading.Lock

    _instance: QwenModel | None = None
    _instance_lock = threading.RLock()

    def __new__(
        cls,
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
        verbose: bool | None = None,
    ) -> QwenModel:
        """Return the one QwenModel, building it on the first call.

        A setting left as None keeps its current value (on the first call,
        the file's). `0` and `False` are real values and still count.

        Asking for a setting that differs from the current one:
        - while prompts are running or queued: raises, the run is not touched;
        - while the server is up but idle: logs a warning, stops the server,
          and the next `ready()` starts it with the new settings;
        - before the server was ever started: just takes the new settings.

        There is no `__init__` on purpose: Python would run it on every
        `QwenModel()` call and reset the model everyone shares.
        """
        asked = {
            "model_path": model_path,
            "server_path": server_path,
            "n_parallel": n_parallel,
            "slot_ctx": slot_ctx,
            "ubatch": ubatch,
            "speculative": speculative,
            "draft_max": draft_max,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking,
            "min_vram_mib": min_vram_mib,
            "verbose": verbose,
        }
        asked = {key: value for key, value in asked.items() if value is not None}

        with QwenModel._instance_lock:
            model = QwenModel._instance

            # First call: build the model from the file, then the arguments.
            if model is None:
                model = super().__new__(cls)
                settings = dict(_FALLBACKS)
                file = config.read_toml(config.QWEN_CONFIG_PATH)
                for table in _MODEL_TABLES:
                    section = file.get(table)
                    if isinstance(section, dict):
                        # Only known keys: a typo in the file is ignored, not fatal.
                        settings.update({k: v for k, v in section.items() if k in _FALLBACKS})
                settings.update(asked)
                for key, value in settings.items():
                    setattr(model, key, value)

                model.log_path = None
                model._proc = None
                model._pool = None
                model._base_url = ""
                model._primed = ""
                model._in_flight = 0
                model._in_flight_lock = threading.Lock()
                QwenModel._instance = model
                return model

            # Later calls: the same model, with any changed settings applied.
            changed = {key: value for key, value in asked.items() if getattr(model, key) != value}
            if not changed:
                return model
            if model._in_flight:
                raise RuntimeError(
                    f"QwenModel: cannot change {sorted(changed)} mid-run -- prompts are "
                    "still running or queued. Wait for the run to finish, then retry."
                )
            if model._proc is not None:
                log.warning(
                    "QwenModel: shutting down the running server to apply %s; "
                    "it starts again with them on the next ready().",
                    changed,
                )
                model.close()
            for key, value in changed.items():
                setattr(model, key, value)
            QwenModel._instance = model
            return model

    # --- lifecycle ----------------------------------------------------------

    def ready(self) -> None:
        """Check the GPU, start the server, check the model talks sense. Idempotent.

        Ready means *loaded and generating language*, not merely loaded. The
        steps run in order, top to bottom; any failure stops the server and
        raises `ModelUnavailable`.
        """
        # A closed model may be reopened, unless another has been built since.
        with QwenModel._instance_lock:
            if QwenModel._instance is None:
                QwenModel._instance = self
            elif QwenModel._instance is not self:
                raise ModelUnavailable(
                    "Refusing to run: this QwenModel was closed and a new one has "
                    "been built since. Use QwenModel() to get the current one."
                )

        # Already running: nothing to do.
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = None

        # 1. Paths are set.
        if not self.model_path:
            raise ModelUnavailable(
                "Refusing to run: no model_path is set. Pass one, or set "
                f"[model].model_path in {config.QWEN_CONFIG_PATH}."
            )
        if not self.server_path or not Path(self.server_path).is_file():
            raise ModelUnavailable(
                f"Refusing to run: no llama-server at {self.server_path!r}. Pass "
                f"server_path, or set [model].server_path in {config.QWEN_CONFIG_PATH}."
            )

        def free_vram_mib() -> int | None:
            """Free VRAM on device 0, or None when nvidia-smi cannot say."""
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            first = out.stdout.strip().splitlines()[:1] if out.returncode == 0 else []
            return int(first[0].strip()) if first and first[0].strip().isdigit() else None

        # 2. The card has room. Checked on every start: it can be taken
        # between one run and the next.
        free_before = free_vram_mib()
        if self.min_vram_mib and free_before is not None and free_before < self.min_vram_mib:
            raise ModelUnavailable(
                _unavailable_message(
                    f"only {free_before} MiB of VRAM free, and the model needs about "
                    f"{self.min_vram_mib} MiB."
                )
            )

        # 3. Start the server on a free local port, logging to a temp file.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._base_url = f"http://127.0.0.1:{port}"
        args = [
            str(self.server_path),
            "-m", str(self.model_path),
            "-ngl", "999",
            "-np", str(self.n_parallel),
            # Split evenly, so every slot holds `slot_ctx` and `fits()`'s
            # check is exact rather than a hope about a shared pool.
            "-c", str(self.n_parallel * self.slot_ctx),
            "--no-kv-unified",
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

        def die_with_parent() -> None:
            """Runs in the server's process: SIGTERM it when Python dies (Linux).

            A server outliving its Python process would keep 18 GB on the card
            with nothing left to stop it.
            """
            PR_SET_PDEATHSIG = 1
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)

        fd, name = tempfile.mkstemp(prefix="qwen-server-", suffix=".log")
        self.log_path = Path(name)
        with os.fdopen(fd, "w") as log_file:
            self._proc = subprocess.Popen(
                args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                # ggml's CUDA init fails on pinned memory on this machine.
                env={**os.environ, "GGML_CUDA_NO_PINNED": "1"},
                preexec_fn=die_with_parent,
            )

        try:
            # 4. Wait until the server answers /health (503 while loading).
            deadline = time.monotonic() + _HEALTH_TIMEOUT_S
            while True:
                if self._proc.poll() is not None:
                    try:
                        tail = "\n".join(self.log_path.read_text(errors="replace").splitlines()[-20:])
                    except OSError:
                        tail = "(no log)"
                    raise ModelUnavailable(
                        _unavailable_message(
                            f"llama-server exited with code {self._proc.returncode} "
                            f"while loading.\n\nLast lines of {self.log_path}:\n{tail}"
                        )
                    )
                try:
                    if self._request("/health", timeout=2).get("status") == "ok":
                        break
                except (OSError, ValueError, ServerError):
                    pass  # not listening yet, or 503 "Loading model"
                if time.monotonic() > deadline:
                    raise ModelUnavailable(
                        f"Refusing to run: llama-server did not become healthy within "
                        f"{_HEALTH_TIMEOUT_S}s. See {self.log_path}."
                    )
                time.sleep(1)

            # 5. The weights are on the GPU. The server does not log offload,
            # so measure it: the load must take most of the weights' size.
            free_after = free_vram_mib()
            weights_mib = os.path.getsize(self.model_path) / 2**20
            if free_before is not None and free_after is not None:
                taken = free_before - free_after
                if taken < _OFFLOAD_SHARE * weights_mib:
                    raise ModelUnavailable(
                        _unavailable_message(
                            f"the load took only {taken} MiB of VRAM for "
                            f"{weights_mib:.0f} MiB of weights, so they are not on the GPU."
                        )
                    )

            # 6. The model talks sense. This model's CUDA path can load cleanly
            # and emit fluent garbage. Deliberately weak -- non-empty, plausible
            # length, mostly words: it catches a broken runtime, not a subtly
            # degraded model.
            reply = self.generate(_HELLO_PROMPT).text
            stripped = reply.strip()
            words = _WORD.findall(stripped)
            letters = sum(len(w) for w in words)
            visible = sum(1 for c in stripped if not c.isspace())
            if not stripped:
                why = "the reply was empty"
            elif len(stripped) > _HELLO_MAX_CHARS:
                why = f"the reply was too long for a greeting ({len(stripped)} chars)"
            elif not words or letters / visible < 0.5:
                why = "the reply had no words in it, or too few to be language"
            else:
                why = None
            if why:
                raise ModelUnavailable(
                    "Refusing to run: the model loaded but its reply to \"Hello\" was "
                    f"not coherent language -- {why}.\n\n"
                    f"It said: {reply[:200]!r}\n\n"
                    "This is the known failure of this model's CUDA path: a clean load\n"
                    "followed by fluent garbage. The usual cause is a stale\n"
                    "libggml-cuda.so surviving a rebuild -- check which one the\n"
                    f"server at {self.server_path} actually loads (ldd)."
                )
        except BaseException:
            self.close()
            raise

        # 7. One worker per slot: the waiting line stays in this process,
        # where `close()` can cancel it.
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, self.n_parallel), thread_name_prefix="qwen-slot"
        )

    def close(self) -> None:
        """Stop the server, free the card, and let go of this model. Idempotent.

        Queued prompts not yet sent are cancelled; their futures raise
        `CancelledError`. The next `QwenModel()` builds a fresh model.
        """
        with QwenModel._instance_lock:
            if QwenModel._instance is self:
                QwenModel._instance = None
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

    # --- generation ---------------------------------------------------------

    def generate(self, prompt: Prompt) -> Generation:
        """Answer one prompt.

        Safe from several threads at once -- the server queues requests into
        its slots -- which is how `submit()`'s workers run. No retry here:
        validation and retry are the controller's.
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
            # The server compiles it to a grammar: ~4x slower decode, the
            # caller's choice by passing a schema or not.
            body["json_schema"] = prompt.schema

        with self._in_flight_lock:
            self._in_flight += 1
        try:
            started = time.monotonic()
            out = self._request("/completion", body)
            latency_ms = int((time.monotonic() - started) * 1000)
        finally:
            with self._in_flight_lock:
                self._in_flight -= 1

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
        """Queue one prompt; at most `n_parallel` run at once, the rest wait.

        A freed slot takes the next queued prompt at once.
        """
        self.ready()
        with self._in_flight_lock:
            self._in_flight += 1

        def finished(_: Future) -> None:
            with self._in_flight_lock:
                self._in_flight -= 1

        future = self._pool.submit(self.generate, prompt)
        future.add_done_callback(finished)
        return future

    def fits(self, prompt: Prompt) -> bool:
        """Can this prompt, its output budget and the repair margin fit one slot?

        One `/tokenize` call, counted in this model's own tokens. llama-server
        answers it on the HTTP thread that received it, from the vocabulary
        alone -- no slot, no GPU -- so it does not wait behind generations and
        is safe to ask just before each submit.
        """
        self.ready()
        size = len(self._request("/tokenize", {"content": self._render(prompt)}).get("tokens") or [])
        return size + (prompt.max_tokens or self.max_tokens) + _REPAIR_MARGIN <= self.slot_ctx

    def cache_shared_prefix(self, sample: Sequence[Prompt]) -> None:
        """Make every slot read the start all prompts share, once, and keep it.

        Every prompt of a run starts with the same system turn (~1,600 of
        ~2,000 tokens); a slot that has read it once reads only the rest. The
        start is worked out from `sample` -- a stream never holds the whole
        set, and every prompt shares the system turn, so its first few are
        enough. Skipped if the slots already hold this prefix.
        """
        self.ready()
        rendered = [self._render(p) for p in sample]

        # The longest string all the samples start with, in characters, cut at
        # the end of a turn so it tokenises exactly as it does inside each
        # prompt. With fewer than two there is nothing shared.
        common = os.path.commonprefix(rendered) if len(rendered) >= 2 else ""
        end = common.rfind(_TURN_END)
        prefix = common[: end + len(_TURN_END)] if end >= 0 else ""
        if prefix and prefix != self._primed:
            body = {"prompt": prefix, "n_predict": 1, "temperature": 0.0, "cache_prompt": True}
            n = max(1, self.n_parallel)
            with ThreadPoolExecutor(max_workers=n) as pool:  # all at once: one per slot
                list(pool.map(lambda _: self._request("/completion", body), range(n)))
            self._primed = prefix

    # --- talking to the server ----------------------------------------------

    def _request(self, path: str, body: dict | None = None, timeout: float = _REQUEST_TIMEOUT_S) -> dict:
        """GET (no body) or POST JSON to the server; the parsed reply.
        possible requests are /health, /tokenize, /completion. The server's reply is JSON.
        A rejected request raises with the server's own message --
        "exceeds the available context", say -- not a bare HTTP code.
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
        """Lay the prompt out as ChatML, with thinking switched off explicitly.

        Built here rather than by the server's chat endpoint, because this
        model's template turns thinking on by default: the empty
        `<think>\\n\\n</think>` is written here, where it is visible and tested.
        """
        header = (
            f"<|im_start|>system\n{prompt.system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt.user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        opener = "<think>\n" if self.enable_thinking else "<think>\n\n</think>\n\n"
        return header + opener
