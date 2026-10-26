"""Claude through Claude Code's headless CLI, on the user's subscription.

**One `claude -p` call per prompt, and that is this file's private business.**
Nothing above knows a CLI exists. No Anthropic API key is used: the CLI runs
on the subscription login, and an `ANTHROPIC_API_KEY` in the environment is
removed from each call so it cannot switch the billing.

**Not one per process.** Unlike `QwenModel` there is no singleton: a CLI call
holds nothing exclusive. `n_parallel` calls run at once, in a pool `ready()`
starts.

**Isolated per call.** Each call runs in a fresh temp folder outside the repo,
deleted afterwards, with no session saved, its own system prompt in place of
Claude Code's, no tools, no MCP servers and no settings. So the model sees
`Prompt.system` and `Prompt.user` and nothing else.

**Not deterministic.** The CLI exposes no temperature, seed or output-token
cap, so `Prompt.max_tokens` and `Prompt.stop` are ignored; a cut-off answer
fails the prompt's `END` check like any failed check. `Prompt.schema` is not
supported.

**Stopping.** A rate limit is waited out; one that outlasts the waits is a
quota and raises `ModelUnavailable`, as do `max_consecutive_failures` failed
calls in a row. A model that raised it stays stopped until `close()`.

**Settings come from `configs/claude.toml`**; an explicit argument wins. See
`docs/specs/0012-claude-codex-model-controller.md`.

Public: `ready`, `generate`, `submit`, `close`. Everything else is private.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from engine import config
from engine.models.base_model import (
    BaseModel,
    Generation,
    ModelUnavailable,
    Prompt,
    Usage,
)

log = logging.getLogger(__name__)

# Used where `configs/claude.toml` says nothing. The binary is empty on
# purpose: a machine-specific path has no sane fallback, so `ready()` refuses.
_FALLBACKS: dict[str, Any] = {
    "binary": "",
    "model": "claude-sonnet-5-5",
    "effort": "medium",
    "n_parallel": 4,
    "timeout_s": 300,
    "rate_limit_wait_s": 60,
    "rate_limit_waits": 3,
    "max_consecutive_failures": 8,
}

# Version and login checks are local; they answer in well under this.
_CHECK_TIMEOUT_S = 30

# What a rate limit says, besides `api_error_status` 429. Not yet seen on a
# real call (spec 0012, Open 1): the consecutive-failure stop is the backstop.
_RATE_LIMIT = re.compile(r"rate.?limit|usage limit|hit your limit|\b429\b", re.IGNORECASE)


class ClaudeModel(BaseModel):
    """Claude Code's `claude -p`, `n_parallel` calls at once.

    Build it with `ClaudeModel(...)` -- free, runs nothing. `ready()` checks
    the CLI; `close()` stops whatever is running.
    """

    # The settings. All come from `configs/claude.toml` unless passed.
    binary: str  # as configured, `~` expanded; calls run `_resolved`
    model: str  # the pinned model id
    effort: str
    n_parallel: int  # calls running at once
    timeout_s: int  # per call
    rate_limit_wait_s: int  # the first wait; each next one doubles
    rate_limit_waits: int
    max_consecutive_failures: int

    def __init__(
        self,
        binary: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        n_parallel: int | None = None,
        timeout_s: int | None = None,
        rate_limit_wait_s: int | None = None,
        rate_limit_waits: int | None = None,
        max_consecutive_failures: int | None = None,
    ):
        """Settings from the file, then the arguments. Runs nothing."""
        asked = {
            "binary": binary,
            "model": model,
            "effort": effort,
            "n_parallel": n_parallel,
            "timeout_s": timeout_s,
            "rate_limit_wait_s": rate_limit_wait_s,
            "rate_limit_waits": rate_limit_waits,
            "max_consecutive_failures": max_consecutive_failures,
        }
        settings = dict(_FALLBACKS)
        section = config.read_toml(config.CLAUDE_CONFIG_PATH).get("model")
        if isinstance(section, dict):
            # Only known keys: a typo in the file is ignored, not fatal.
            settings.update({k: v for k, v in section.items() if k in _FALLBACKS})
        settings.update({k: v for k, v in asked.items() if v is not None})
        settings["binary"] = os.path.expanduser(settings["binary"]) if settings["binary"] else ""
        for key, value in settings.items():
            setattr(self, key, value)

        self._resolved = ""  # the binary's real file, fixed by ready()
        self._version = ""  # the CLI's version, read by ready()
        self._pool: ThreadPoolExecutor | None = None
        self._running: set[Any] = set()  # call processes, for close() to kill
        self._failures = 0  # failed calls in a row
        self._stopped = ""  # why the model raised ModelUnavailable; "" while fit
        self._lock = threading.RLock()

    # --- lifecycle ----------------------------------------------------------

    def ready(self) -> None:
        """Check the CLI runs and is logged in; start the pool. Idempotent.

        No model call: one would spend subscription quota on every start.
        Any failure raises `ModelUnavailable` naming the fix.
        """
        with self._lock:
            if self._stopped:
                raise ModelUnavailable(self._stopped)
            if self._pool is not None:
                return

            # 1. The binary exists. The symlink is resolved once, so every call
            # of this run uses one version even if Claude Code updates itself.
            resolved = Path(self.binary).resolve() if self.binary else None
            if resolved is None or not resolved.is_file():
                raise ModelUnavailable(
                    f"Refusing to run: no Claude Code binary at {self.binary!r}. Pass "
                    f"binary, or set [model].binary in {config.CLAUDE_CONFIG_PATH}."
                )

            # 2. It runs, and says which version it is.
            try:
                out = subprocess.run(
                    [str(resolved), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=_CHECK_TIMEOUT_S,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ModelUnavailable(
                    f"Refusing to run: {resolved} --version failed ({exc})."
                ) from exc
            if out.returncode != 0 or not out.stdout.strip():
                raise ModelUnavailable(
                    f"Refusing to run: {resolved} --version exited with code "
                    f"{out.returncode}: {out.stderr.strip()[:200]}"
                )
            version = out.stdout.split()[0]

            # 3. It is logged in, on the subscription.
            try:
                out = subprocess.run(
                    [str(resolved), "auth", "status"],
                    capture_output=True,
                    text=True,
                    timeout=_CHECK_TIMEOUT_S,
                )
                logged_in = json.loads(out.stdout or "{}").get("loggedIn") is True
            except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
                logged_in = False
            if not logged_in:
                raise ModelUnavailable(
                    "Refusing to run: Claude Code is not logged in. Run `claude login` "
                    "and choose the claude.ai subscription."
                )

            self._resolved = str(resolved)
            self._version = version
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, self.n_parallel), thread_name_prefix="claude-call"
            )

    def close(self) -> None:
        """Cancel queued prompts, kill running calls, clear a stop. Idempotent.

        Kills rather than waits: a call can run for `timeout_s`, and an
        interrupted notebook should not hang that long. Killed calls' answers
        are discarded. `ready()` after `close()` makes the model usable again.
        """
        with self._lock:
            pool, self._pool = self._pool, None
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            for proc in list(self._running):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass  # already gone
            self._failures = 0
            self._stopped = ""

    # --- generation ---------------------------------------------------------

    def generate(self, prompt: Prompt) -> Generation:
        """Answer one prompt with one CLI call.

        Safe from several threads at once, which is how `submit()`'s workers
        run. No retry of a failed answer here: that is the controller's.

        A failed call raises one of two things. `ModelUnavailable` when the
        model must stop: a rate limit past the last wait, or the
        `max_consecutive_failures`-th failed call in a row. Anything else is
        that call's own error, and the next call may well succeed.
        """
        self.ready()
        session = self._pool  # this ready()'s pool; close() ends it
        try:
            with tempfile.TemporaryDirectory(prefix="claude-call-") as temp:
                # 1. The call's own folder: the system prompt as a file, nothing else.
                system_file = Path(temp) / "system.md"
                system_file.write_text(prompt.system)
                args = [
                    self._resolved,
                    "-p",
                    "--model", self.model,
                    "--effort", self.effort,
                    # Replaces Claude Code's own system prompt.
                    "--system-prompt-file", str(system_file),
                    "--tools", "",
                    "--disable-slash-commands",
                    # Without these two every call loaded the user's connectors and
                    # settings: 202,253 input tokens for a ~1,600-token prompt.
                    "--strict-mcp-config",
                    "--setting-sources", "",
                    "--no-session-persistence",
                    "--output-format", "json",
                    # Never --bare: it reads auth only from an API key.
                ]
                env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

                # 2. Run it; a rate limit is waited out and the call re-run.
                waits = 0
                while True:
                    with self._lock:
                        # Closed meanwhile -- during a rate-limit wait, say: start
                        # nothing close() would not know to kill. Started and
                        # registered under one lock, so close() never misses one.
                        if self._pool is not session:
                            raise CancelledError()
                        if self._stopped:  # another call stopped the model meanwhile
                            raise ModelUnavailable(self._stopped)
                        started = time.monotonic()
                        proc = subprocess.Popen(
                            args,
                            cwd=temp,
                            env=env,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            start_new_session=True,  # one group, killed as one
                        )
                        self._running.add(proc)
                    try:
                        stdout, stderr = proc.communicate(prompt.user, timeout=self.timeout_s)
                    except subprocess.TimeoutExpired:
                        # The CLI is Node with children of its own: kill the group,
                        # so nothing keeps running and spending quota.
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                        proc.communicate()
                        raise RuntimeError(f"claude timed out after {self.timeout_s}s")
                    finally:
                        with self._lock:
                            self._running.discard(proc)
                    latency_ms = int((time.monotonic() - started) * 1000)

                    # 3. Did it answer? The JSON says why not, when it is there.
                    try:
                        out = json.loads(stdout)
                    except ValueError:
                        out = None
                    if not isinstance(out, dict):
                        raise RuntimeError(
                            f"claude exited with code {proc.returncode} and no JSON answer: "
                            f"{(stdout or stderr).strip()[:300]}"
                        )
                    if not out.get("is_error") and proc.returncode == 0:
                        break
                    message = str(out.get("result") or stderr).strip()[:300]
                    failure = (
                        f"claude exited with code {proc.returncode} ({out.get('subtype')}, "
                        f"api_error_status {out.get('api_error_status')}): {message}"
                    )
                    rate_limited = out.get("api_error_status") == 429 or _RATE_LIMIT.search(message)
                    if not rate_limited:
                        raise RuntimeError(failure)
                    if waits < self.rate_limit_waits:
                        wait_s = self.rate_limit_wait_s * 2**waits
                        log.warning("claude rate limit; waiting %ss, then re-running: %s", wait_s, message)
                        time.sleep(wait_s)
                        waits += 1
                        continue
                    # Still limited after every wait: a quota, which resets in
                    # hours. Retrying it per prompt would only fail the rest of
                    # the run; stopping leaves it resumable.
                    with self._lock:
                        self._stopped = (
                            f"Refusing to run: claude is still rate limited after {waits} "
                            f"waits ({failure}). The quota resets later; close() the "
                            "model and resume the run then."
                        )
                    raise ModelUnavailable(self._stopped)

                # 4. Read the answer: the pinned model's, or none.
                usage = (out.get("modelUsage") or {}).get(self.model)
                if not isinstance(usage, dict):
                    raise RuntimeError(
                        f"claude answered without the pinned model {self.model!r}; "
                        f"modelUsage names {sorted(out.get('modelUsage') or {})}"
                    )
                generation = Generation(
                    text=str(out.get("result") or ""),
                    latency_ms=latency_ms,
                    usage=Usage(
                        prompt_tokens=int(usage.get("inputTokens") or 0)
                        + int(usage.get("cacheCreationInputTokens") or 0)
                        + int(usage.get("cacheReadInputTokens") or 0),
                        completion_tokens=int(usage.get("outputTokens") or 0),
                    ),
                    stop_reason=str(out.get("stop_reason") or ""),
                    model=f"{self.model}@claude-code-{self._version}",
                )
        except ModelUnavailable:
            raise
        except Exception as exc:
            # 5. Count it. A run of failures with no success between them is
            # most likely one cause for all -- a quota whose signal is not
            # recognised, the login gone -- so the model stops. A call killed
            # by close() is not the CLI's failure and is not counted, nor is one
            # from before the last close().
            with self._lock:
                if self._pool is session:
                    self._failures += 1
                    if self._failures >= self.max_consecutive_failures:
                        self._stopped = (
                            f"Refusing to run: {self._failures} calls in a row failed; the "
                            f"last: {exc}. close() the model and ready() it once the cause "
                            "is fixed."
                        )
                        raise ModelUnavailable(self._stopped) from exc
            raise
        with self._lock:
            if self._pool is session:
                self._failures = 0
        return generation

    def submit(self, prompt: Prompt) -> Future[Generation]:
        """Queue one prompt; at most `n_parallel` calls run at once, the rest wait."""
        with self._lock:  # so a close() in between cannot pull the pool away
            self.ready()
            return self._pool.submit(self.generate, prompt)
