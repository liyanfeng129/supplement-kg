"""Qwen3-Embedding on llama.cpp's servers: one shared model, GPU-only, many texts at once.

**One model per process.** `EmbedModel()` always returns the same object, so
the weights are never loaded twice. Asking for different settings restarts the
server with them -- but only while nothing is running.

**Hosted by `llama-server`, and that is this file's private business.**
`ready()` starts the servers, the model talks to them over HTTP on localhost,
and `close()` stops them. The start, health and stop logic is copied from
`qwen_model.py`, not shared: a common helper is a later refactor, once two
working copies show what is really common.

**Parallelism is servers, not slots.** The model runs `n_servers` servers on
one card, each loading the weights and running one slot: one slot was the
fastest of every count measured, and three single-slot servers together are
1.68x one (`configs/embed.toml`). Each request goes to the server with the
fewest requests in flight. Any server failing makes the whole model
unavailable; none is restarted or dropped. A text gets the same vector from
any server.

**GPU-only, and the vectors must mean something.** `ready()` refuses a card
without room and a load that did not reach the GPU, as Qwen does; and because a
wrong pooling mode, a missing end token or a broken template give well-formed
vectors that mean nothing, with no error, it checks that near-identical texts
are closer than unrelated ones.

**The instruction template is this model's**, as ChatML is Qwen's: the caller
passes its wording, and the text sent is `Instruct: {instruction}\\nQuery:{text}`.

**Settings come from `configs/embed.toml`**; an explicit argument wins. See
`docs/specs/0007-embed-model-controller.md` and
`docs/specs/0008-label-embedding-service.md`.

Public: `ready`, `embed`, `fits`, `submit`, `close`. Everything else is private.
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

import numpy as np

from engine import config
from engine.models.base_model import BaseEmbedModel, Embedding, ModelUnavailable, Usage

log = logging.getLogger(__name__)

# Used where `configs/embed.toml` says nothing. The paths are empty on purpose:
# a machine-specific path has no sane fallback, so `ready()` refuses instead.
_FALLBACKS: dict[str, Any] = {
    "model_path": "",
    "server_path": "",
    "n_servers": 1,
    "n_ctx": 32768,
    "ubatch": 8192,
    "min_vram_mib": 8000,
}

# Qwen3-Embedding-0.6B's vector size. A different model changes this and
# `embed.toml`, nothing else.
_DIM = 1024

# What last-token pooling expects at the end of every text.
_END_TOKEN = "<|endoftext|>"

# A row whose length is further than this from 1 is not L2-normalised.
_UNIT_TOLERANCE = 1e-3

# A load taking less VRAM than this share of the weights file did not put the
# weights on the GPU (a CPU fallback takes a few hundred MiB).
_OFFLOAD_SHARE = 0.9

_HEALTH_TIMEOUT_S = 900
_REQUEST_TIMEOUT_S = 1800

# The similarity check: two near-identical supplement passages and one
# unrelated text. Weak on purpose -- it catches broken pooling, a missing end
# token or a broken template, not a subtly worse model.
_NEAR_A = (
    "Vitamin D3 2000 IU softgels. Supports bone health and immune function. "
    "Take one softgel daily with food."
)
_NEAR_B = (
    "Vitamin D3 2000 IU softgel. Supports healthy bones and immune function. "
    "Take one softgel per day with a meal."
)
_UNRELATED = (
    "Stainless steel garden hose nozzle with eight spray patterns and a "
    "comfortable non-slip grip."
)
_TEST_INSTRUCTION = "Identify the dietary supplement product described by this text"


class ServerError(RuntimeError):
    """The server answered with an error; the message is the server's own."""


def _unavailable_message(reason: str) -> str:
    """Explain why the GPU is unusable, and name whatever process is holding it."""
    lines = [
        f"Refusing to run: {reason}",
        "",
        "This model is GPU-only by design. Falling back to CPU is not an",
        "option here -- it is far slower, and llama.cpp does it silently:",
        "no error, correct vectors, and only the clock to tell you.",
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
            "kernel with a model loaded counts, and so does the Qwen model: the",
            "two do not fit on the card together. Stop it and retry:",
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


class EmbedModel(BaseEmbedModel):
    """The one embedding model: `n_servers` single-slot servers, many texts.

    Build it with `EmbedModel(...)` -- free, touches no GPU. `ready()` starts
    the servers; `close()` stops them.
    """

    # The settings. All come from `configs/embed.toml` unless passed.
    model_path: str
    server_path: str
    n_servers: int  # servers on the card, one slot each: texts embedded at once
    n_ctx: int  # each server's one slot's KV
    ubatch: int  # one forward pass; also the most tokens one text may have
    min_vram_mib: int

    log_paths: list[Path]  # each server's log file, in start order

    # Private state, set up in `__new__`. The three lists are per server, in
    # start order.
    _procs: list[Any]  # the llama-server processes
    _base_urls: list[str]
    _busy: list[int]  # requests being answered by each server
    _pool: ThreadPoolExecutor | None  # one worker per server, created by ready()
    _end_token: str  # appended to every text: "" when the tokenizer adds it itself
    _in_flight: int  # requests queued or being answered, all servers together
    _in_flight_lock: threading.Lock  # guards `_in_flight` and `_busy`

    _instance: EmbedModel | None = None
    _instance_lock = threading.RLock()

    def __new__(
        cls,
        model_path: str | None = None,
        server_path: str | None = None,
        n_servers: int | None = None,
        n_ctx: int | None = None,
        ubatch: int | None = None,
        min_vram_mib: int | None = None,
    ) -> EmbedModel:
        """Return the one EmbedModel, building it on the first call.

        As `QwenModel()`: a setting left as None keeps its current value (on
        the first call, the file's). A changed setting is refused while
        requests are running or queued, stops an idle server so the next
        `ready()` starts it with the new settings, and is simply taken before
        the server was ever started.
        """
        asked = {
            "model_path": model_path,
            "server_path": server_path,
            "n_servers": n_servers,
            "n_ctx": n_ctx,
            "ubatch": ubatch,
            "min_vram_mib": min_vram_mib,
        }
        asked = {key: value for key, value in asked.items() if value is not None}

        with EmbedModel._instance_lock:
            model = EmbedModel._instance

            # First call: build the model from the file, then the arguments.
            if model is None:
                model = super().__new__(cls)
                settings = dict(_FALLBACKS)
                # `[model]` only: `[controller]` is the controller's.
                section = config.read_toml(config.EMBED_CONFIG_PATH).get("model")
                if isinstance(section, dict):
                    # Only known keys: a typo in the file is ignored, not fatal.
                    settings.update({k: v for k, v in section.items() if k in _FALLBACKS})
                settings.update(asked)
                for key, value in settings.items():
                    setattr(model, key, value)

                model.log_paths = []
                model._procs = []
                model._base_urls = []
                model._busy = []
                model._pool = None
                model._end_token = ""
                model._in_flight = 0
                model._in_flight_lock = threading.Lock()
                EmbedModel._instance = model
                return model

            # Later calls: the same model, with any changed settings applied.
            changed = {key: value for key, value in asked.items() if getattr(model, key) != value}
            if not changed:
                return model
            if model._in_flight:
                raise RuntimeError(
                    f"EmbedModel: cannot change {sorted(changed)} mid-run -- requests are "
                    "still running or queued. Wait for the run to finish, then retry."
                )
            if model._procs:
                log.warning(
                    "EmbedModel: shutting down the running servers to apply %s; "
                    "it starts again with them on the next ready().",
                    changed,
                )
                model.close()
            for key, value in changed.items():
                setattr(model, key, value)
            EmbedModel._instance = model
            return model

    # --- lifecycle ----------------------------------------------------------

    def ready(self) -> None:
        """Check the GPU, start every server, settle the end token, check the vectors. Idempotent.

        Ready means *every server loaded and giving vectors that mean
        something*, not merely loaded. The servers start one after another,
        each through steps 3 to 8 in order; any failure stops every server
        started so far and raises `ModelUnavailable`.
        """
        # A closed model may be reopened, unless another has been built since.
        with EmbedModel._instance_lock:
            if EmbedModel._instance is None:
                EmbedModel._instance = self
            elif EmbedModel._instance is not self:
                raise ModelUnavailable(
                    "Refusing to run: this EmbedModel was closed and a new one has "
                    "been built since. Use EmbedModel() to get the current one."
                )

        # Already running: nothing to do. A server this model started that has
        # since exited is not restarted, and the others are not run without
        # it: whatever killed it would likely kill the next one, and a person
        # decides.
        if self._procs:
            for k, proc in enumerate(self._procs):
                if proc.poll() is not None:
                    raise ModelUnavailable(
                        f"Refusing to run: llama-server {k + 1} of {len(self._procs)} exited "
                        f"with code {proc.returncode} mid-run, and the model is not restarting "
                        f"it. See {self.log_paths[k]}. close() the model, then ready() it again "
                        "to start fresh servers."
                    )
            return

        # 1. Paths are set.
        if not self.model_path:
            raise ModelUnavailable(
                "Refusing to run: no model_path is set. Pass one, or set "
                f"[model].model_path in {config.EMBED_CONFIG_PATH}."
            )
        if not self.server_path or not Path(self.server_path).is_file():
            raise ModelUnavailable(
                f"Refusing to run: no llama-server at {self.server_path!r}. Pass "
                f"server_path, or set [model].server_path in {config.EMBED_CONFIG_PATH}."
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

        # 2. The card has room for every server, checked once before any of
        # them starts, and again on every later ready(): the card can be taken
        # between one run and the next.
        free_before = free_vram_mib()
        needed = self.n_servers * self.min_vram_mib
        if self.min_vram_mib and free_before is not None and free_before < needed:
            raise ModelUnavailable(
                _unavailable_message(
                    f"only {free_before} MiB of VRAM free, and {self.n_servers} servers need "
                    f"about {self.min_vram_mib} MiB each ({needed} MiB)."
                )
            )

        def die_with_parent() -> None:
            """Runs in the server's process: SIGTERM it when Python dies (Linux).

            A server outliving its Python process would keep the card with
            nothing left to stop it.
            """
            PR_SET_PDEATHSIG = 1
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)

        # Every server started so far, in start order. While one is being
        # checked, the model's own lists hold that server alone, so `embed`
        # and `_request` reach it and nothing else.
        procs: list[Any] = []
        base_urls: list[str] = []
        self.log_paths = []
        try:
            for k in range(self.n_servers):
                which = f"llama-server {k + 1} of {self.n_servers}"
                if k:
                    free_before = free_vram_mib()

                # 3. Start the server on a free local port, logging to a temp file.
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    port = s.getsockname()[1]
                base_url = f"http://127.0.0.1:{port}"
                args = [
                    str(self.server_path),
                    "-m", str(self.model_path),
                    "--embedding",
                    # What Qwen3-Embedding is trained for. A wrong mode gives
                    # well-formed vectors that mean nothing; step 8 catches it.
                    "--pooling", "last",
                    # L2, the server's default, passed so unit length does not
                    # rest on a default. Step 7 confirms it; nothing
                    # re-normalises in Python.
                    "--embd-normalize", "2",
                    "-ngl", "999",
                    # One slot: the fastest count measured. It has no KV to
                    # share, so `n_ctx` is its own.
                    "-np", "1",
                    "-c", str(self.n_ctx),
                    # A whole text must fit in one pass, so `ubatch` is the per-text limit.
                    "-ub", str(self.ubatch),
                    "-b", str(self.ubatch),
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "--no-webui",
                ]
                fd, name = tempfile.mkstemp(prefix=f"embed-server-{k}-", suffix=".log")
                log_path = Path(name)
                self.log_paths.append(log_path)
                with os.fdopen(fd, "w") as log_file:
                    proc = subprocess.Popen(
                        args,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        # ggml's CUDA init fails on pinned memory on this machine.
                        env={**os.environ, "GGML_CUDA_NO_PINNED": "1"},
                        preexec_fn=die_with_parent,
                    )
                procs.append(proc)
                base_urls.append(base_url)
                self._procs, self._base_urls, self._busy = [proc], [base_url], [0]

                # 4. Wait until the server answers /health (503 while loading).
                deadline = time.monotonic() + _HEALTH_TIMEOUT_S
                while True:
                    if proc.poll() is not None:
                        try:
                            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
                        except OSError:
                            tail = "(no log)"
                        raise ModelUnavailable(
                            _unavailable_message(
                                f"{which} exited with code {proc.returncode} "
                                f"while loading.\n\nLast lines of {log_path}:\n{tail}"
                            )
                        )
                    try:
                        if self._request("/health", timeout=2).get("status") == "ok":
                            break
                    except (OSError, ValueError, ServerError):
                        pass  # not listening yet, or 503 "Loading model"
                    if time.monotonic() > deadline:
                        raise ModelUnavailable(
                            f"Refusing to run: {which} did not become healthy within "
                            f"{_HEALTH_TIMEOUT_S}s. See {log_path}."
                        )
                    time.sleep(1)

                # 5. Settle the end token, on the first server: every server
                # loads the same file. Last-token pooling expects it at the end
                # of every text, and the server appends one only when the GGUF
                # sets `add_eos_token`. Tokenise a probe the way the server
                # tokenises an embedding input; if it does not end in the end
                # token, `embed` appends it from now on.
                if k == 0:
                    end = self._request(
                        "/tokenize", {"content": _END_TOKEN, "add_special": False, "parse_special": True}
                    ).get("tokens") or []
                    if len(end) != 1:
                        raise ModelUnavailable(
                            f"Refusing to run: {_END_TOKEN} is not one token in this model's "
                            f"vocabulary ({end!r}), so this is not a Qwen3-Embedding model."
                        )
                    probe = self._request(
                        "/tokenize", {"content": "probe", "add_special": True, "parse_special": True}
                    ).get("tokens") or []
                    self._end_token = "" if probe[-1:] == end else _END_TOKEN

                # 6. This server's weights are on the GPU. The server does not
                # log offload, so measure it: its load must take most of the
                # weights' size.
                free_after = free_vram_mib()
                weights_mib = os.path.getsize(self.model_path) / 2**20
                if free_before is not None and free_after is not None:
                    taken = free_before - free_after
                    if taken < _OFFLOAD_SHARE * weights_mib:
                        raise ModelUnavailable(
                            _unavailable_message(
                                f"{which} took only {taken} MiB of VRAM for "
                                f"{weights_mib:.0f} MiB of weights, so they are not on the GPU."
                            )
                        )

                # 7. Vectors are well-formed: one row, the right size, unit
                # length, no NaN.
                vectors = self.embed([_NEAR_A]).vectors
                norms = np.linalg.norm(vectors, axis=1)
                if vectors.shape != (1, _DIM):
                    why = f"one text gave vectors of shape {vectors.shape}, not (1, {_DIM})"
                elif np.isnan(vectors).any():
                    why = "the vector has NaN in it"
                elif abs(norms[0] - 1.0) > _UNIT_TOLERANCE:
                    why = f"the vector is not unit length (norm {norms[0]:.4f})"
                else:
                    why = None
                if why:
                    raise ModelUnavailable(
                        f"Refusing to run: {which} loaded but its vectors are malformed -- {why}."
                    )

                # 8. Similarity is ordered sensibly, raw and with an
                # instruction: the near-identical pair is closer than either is
                # to the unrelated text. Rows are unit length, so a dot product
                # is the cosine.
                for instruction in (None, _TEST_INSTRUCTION):
                    a, b, c = self.embed([_NEAR_A, _NEAR_B, _UNRELATED], instruction).vectors
                    ab, ac, bc = float(a @ b), float(a @ c), float(b @ c)
                    if not (ab > ac and ab > bc):
                        how = "without an instruction" if instruction is None else "with an instruction"
                        raise ModelUnavailable(
                            f"Refusing to run: {which} loaded but two near-identical "
                            f"passages are not more similar to each other ({ab:.3f}) than to an "
                            f"unrelated text ({ac:.3f}, {bc:.3f}), {how}. The usual causes "
                            "are the wrong pooling mode, a missing end token, or a broken "
                            f"instruction template. See {log_path}."
                        )
        except BaseException:
            self._procs = procs
            self.close()
            raise

        # 9. Every server is checked: requests may now go to any of them. One
        # worker per server: the waiting line stays in this process, where
        # `close()` can cancel it.
        self._procs, self._base_urls, self._busy = procs, base_urls, [0] * len(procs)
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, self.n_servers), thread_name_prefix="embed-server"
        )

    def close(self) -> None:
        """Stop every server, free the card, and let go of this model. Idempotent.

        Queued requests not yet sent are cancelled; their futures raise
        `CancelledError`. The next `EmbedModel()` builds a fresh model.
        """
        with EmbedModel._instance_lock:
            if EmbedModel._instance is self:
                EmbedModel._instance = None
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        procs, self._procs = self._procs, []
        self._base_urls, self._busy = [], []
        # Signal every server first, then wait: they stop side by side.
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    # --- embedding ----------------------------------------------------------

    def embed(self, texts: Sequence[str], instruction: str | None = None) -> Embedding:
        """One unit vector per text, rows in input order, in one request.

        Safe from several threads at once, which is how `submit()`'s workers
        run: each request goes to the server with the fewest requests in
        flight (ties: the first). No retry here: retry is the controller's.

        A failed request raises one of two things. `ModelUnavailable` when a
        server is gone -- nothing more can run. Anything else is that
        request's own error, and the servers are still fit for the next.
        """
        self.ready()
        # The model's template, then the end token if the tokenizer does not
        # add it.
        templated = [
            (t if instruction is None else f"Instruct: {instruction}\nQuery:{t}") + self._end_token
            for t in texts
        ]
        body = {"input": templated, "encoding_format": "float"}

        # The least-busy server takes it.
        with self._in_flight_lock:
            server = min(range(len(self._busy)), key=self._busy.__getitem__)
            self._busy[server] += 1
            self._in_flight += 1
        try:
            started = time.monotonic()
            out = self._request("/v1/embeddings", body, server=server)
            latency_ms = int((time.monotonic() - started) * 1000)
        except Exception as exc:
            # This request's problem, or the model's? As Qwen: the model is
            # gone when any server's process has exited, when this one reports
            # a CUDA or out-of-memory failure, or when it no longer answers
            # /health ok. Otherwise the error is this request's own, and the
            # controller retries it.
            procs = self._procs
            exited = [k for k, proc in enumerate(procs) if proc.poll() is not None]
            if not procs:
                failure_reason = "the model was closed"
            elif exited:
                failure_reason = "; ".join(
                    f"llama-server {k + 1} exited with code {procs[k].returncode}" for k in exited
                )
            elif re.search(r"CUDA|out of memory", str(exc), re.IGNORECASE):
                failure_reason = f"llama-server {server + 1} reported a CUDA or out-of-memory failure"
            else:
                try:
                    healthy = self._request("/health", timeout=5, server=server).get("status") == "ok"
                except (OSError, ValueError, ServerError):
                    healthy = False
                failure_reason = None if healthy else f"llama-server {server + 1} no longer answers /health"
            if failure_reason:
                raise ModelUnavailable(
                    f"{failure_reason} during a request ({exc}). The model is not restarting "
                    f"it. See {self.log_paths[server]}."
                ) from exc
            raise
        finally:
            with self._in_flight_lock:
                if server < len(self._busy):
                    self._busy[server] -= 1
                self._in_flight -= 1

        # The server may answer in any order; each row carries its index.
        rows = sorted(out.get("data") or [], key=lambda row: row["index"])
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        return Embedding(
            vectors=vectors.reshape(len(rows), -1),
            latency_ms=latency_ms,
            usage=Usage(prompt_tokens=int((out.get("usage") or {}).get("prompt_tokens") or 0)),
        )

    def submit(self, texts: Sequence[str], instruction: str | None = None) -> Future[Embedding]:
        """Queue one request; at most `n_servers` run at once, the rest wait.

        Never blocks, never refuses.
        """
        self.ready()
        with self._in_flight_lock:
            self._in_flight += 1

        def finished(_: Future) -> None:
            with self._in_flight_lock:
                self._in_flight -= 1

        future = self._pool.submit(self.embed, list(texts), instruction)
        future.add_done_callback(finished)
        return future

    def fits(self, text: str, instruction: str | None = None) -> bool:
        """Can this text, templated and with its end token, run in one pass?

        A token covers at least one character, so a templated text with no
        more characters than `ubatch` less the end token fits without asking.
        Otherwise one `/tokenize` call to the first server counts it in this
        model's own tokens, the end token included -- every server loads the
        same vocabulary; llama-server answers it from the vocabulary alone, so
        it does not wait behind embeddings.
        """
        templated = text if instruction is None else f"Instruct: {instruction}\nQuery:{text}"
        if len(templated) + 1 <= self.ubatch:
            return True
        self.ready()
        tokens = self._request(
            "/tokenize",
            {"content": templated + self._end_token, "add_special": True, "parse_special": True},
        ).get("tokens") or []
        return len(tokens) <= self.ubatch

    # --- talking to the servers ---------------------------------------------

    def _request(
        self, path: str, body: dict | None = None, timeout: float = _REQUEST_TIMEOUT_S, server: int = 0
    ) -> dict:
        """GET (no body) or POST JSON to server `server`; the parsed reply.

        The requests are /health, /tokenize and /v1/embeddings. A rejected
        request raises with the server's own message -- "input is too large
        to process", say -- not a bare HTTP code.
        """
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self._base_urls[server] + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise ServerError(f"llama-server {path} -> {exc.code}: {detail}") from exc
