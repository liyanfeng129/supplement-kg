"""ClaudeModel: Claude Code's headless CLI on the subscription, one call per prompt.

Pinned here: settings come from `configs/claude.toml` with explicit arguments
winning; `ready()` resolves the binary once and refuses one that is missing
or not logged in, naming the fix; `generate()` runs one stripped, isolated
`claude -p` call and reads its JSON into a `Generation`; a rate limit is
waited out and, past the last wait, stops the model; failed calls are the
request's own error until `max_consecutive_failures` in a row stop the model;
a stopped model stays stopped until `close()`; a timeout and `close()` kill
the call's process group; `submit()` runs at most `n_parallel` at once.

Only public functions are called. No CLI runs: underneath the model,
`subprocess.run` (version and login), `subprocess.Popen` (the call),
`os.killpg` and `time.sleep` are replaced.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time

import pytest

from engine.models import claude_model as cm
from engine.models.base_model import ModelUnavailable, Prompt, Usage
from engine.models.claude_model import ClaudeModel

PROMPT = Prompt(system="Extract claims.", user="supports healthy joints.")


def answer(text="P: supports healthy joints\nEND", **overrides) -> str:
    """The JSON object `claude -p --output-format json` prints, as measured."""
    out = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "api_error_status": None,
        "stop_reason": "end_turn",
        "result": text,
        "modelUsage": {
            "claude-haiku-5-5": {"inputTokens": 1190, "outputTokens": 17},
            "claude-sonnet-5-5": {
                "inputTokens": 2,
                "outputTokens": 24,
                "cacheReadInputTokens": 100,
                "cacheCreationInputTokens": 1619,
            },
        },
    }
    out.update(overrides)
    return json.dumps(out)


class FakeCli:
    """Stands in for the CLI: `subprocess.run` for checks, `Popen` for calls.

    `calls` is a script: each entry is `(stdout, returncode)`, `"timeout"`
    for a call that never finishes, or `"block"` for one that runs until its
    process group is killed. What each call was given is
    kept in `seen` as `(args, kwargs, stdin, files in its cwd)`.
    """

    def __init__(self, calls=(), version="2.1.310 (Claude Code)", logged_in=True, version_rc=0):
        self.calls = list(calls)
        self.version = version
        self.logged_in = logged_in
        self.version_rc = version_rc
        self.run_args: list[list[str]] = []
        self.seen: list[tuple] = []
        self.killed: list[int] = []
        self.slept: list[float] = []
        self.lock = threading.Lock()
        self.released = threading.Event()  # set by the first killpg
        self.running = 0
        self.started = 0  # processes started, which numbers their pids
        self.most_running = 0

    def run(self, args, **kwargs):
        self.run_args.append(list(args))
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, self.version_rc, self.version + "\n", "")
        if args[1:] == ["auth", "status"]:
            status = {"loggedIn": self.logged_in, "authMethod": "claude.ai"}
            return subprocess.CompletedProcess(args, 0 if self.logged_in else 1, json.dumps(status), "")
        raise AssertionError(f"unexpected run {args}")

    def popen(self, args, **kwargs):
        cli = self
        with self.lock:
            pid = 4242 + self.started
            self.started += 1

        class Proc:
            returncode = None

            def __init__(self):
                self.pid = pid

            def communicate(self, input=None, timeout=None):
                if input is None:  # the reap after a kill
                    return "", ""
                cwd = kwargs["cwd"]
                files = {name: open(os.path.join(cwd, name)).read() for name in os.listdir(cwd)}
                with cli.lock:
                    cli.seen.append((list(args), kwargs, input, files))
                    step = cli.calls.pop(0)
                if step == "timeout":
                    raise subprocess.TimeoutExpired(args, timeout)
                if step == "block":
                    with cli.lock:
                        cli.running += 1
                        cli.most_running = max(cli.most_running, cli.running)
                    cli.released.wait(5)
                    with cli.lock:
                        cli.running -= 1
                    self.returncode = -9
                    return "", "killed"
                stdout, self.returncode = step
                return stdout, "stderr text"

        return Proc()

    def killpg(self, pid, sig):
        assert sig == signal.SIGKILL
        self.killed.append(pid)
        self.released.set()


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """A fake CLI, and a real binary path that is a symlink to a versioned file."""
    versions = tmp_path / "versions"
    versions.mkdir()
    (versions / "2.1.310").write_text("")
    (tmp_path / "claude").symlink_to(versions / "2.1.310")
    fake = FakeCli()
    monkeypatch.setattr(subprocess, "run", fake.run)
    monkeypatch.setattr(subprocess, "Popen", fake.popen)
    monkeypatch.setattr(os, "killpg", fake.killpg)
    monkeypatch.setattr(time, "sleep", fake.slept.append)
    fake.binary = str(tmp_path / "claude")
    fake.resolved = str(versions / "2.1.310")
    return fake


def model(cli, **kwargs) -> ClaudeModel:
    return ClaudeModel(binary=cli.binary, **kwargs)


# --- settings ---------------------------------------------------------------


def test_the_committed_file_supplies_the_settings():
    m = ClaudeModel()

    assert m.binary == os.path.expanduser("~/.local/bin/claude")
    assert (m.model, m.effort) == ("claude-sonnet-5-5", "medium")
    assert (m.n_parallel, m.timeout_s) == (4, 300)
    assert (m.rate_limit_wait_s, m.rate_limit_waits, m.max_consecutive_failures) == (60, 3, 8)


def test_an_explicit_argument_beats_the_file():
    assert ClaudeModel(effort="high", n_parallel=2).effort == "high"
    assert ClaudeModel(n_parallel=2).n_parallel == 2


def test_a_missing_file_falls_back_to_the_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(cm.config, "CLAUDE_CONFIG_PATH", tmp_path / "missing.toml")

    m = ClaudeModel()

    assert (m.binary, m.model, m.n_parallel) == ("", "claude-sonnet-5-5", 4)


# --- ready ------------------------------------------------------------------


def test_ready_resolves_the_binary_once_and_reads_its_version(cli):
    m = model(cli)

    m.ready()
    m.ready()

    assert cli.run_args == [[cli.resolved, "--version"], [cli.resolved, "auth", "status"]]
    m.close()


def test_ready_refuses_a_binary_that_does_not_exist(cli, tmp_path):
    m = ClaudeModel(binary=str(tmp_path / "nowhere"))

    with pytest.raises(ModelUnavailable, match="binary"):
        m.ready()


def test_ready_refuses_a_binary_that_does_not_run(cli):
    cli.version_rc = 127

    with pytest.raises(ModelUnavailable, match="--version"):
        model(cli).ready()


def test_ready_refuses_a_cli_that_is_not_logged_in(cli):
    cli.logged_in = False

    with pytest.raises(ModelUnavailable, match="claude login"):
        model(cli).ready()


# --- generate ---------------------------------------------------------------


def test_one_call_is_stripped_isolated_and_read_into_a_generation(cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-reach-the-cli")
    cli.calls = [(answer(), 0)]
    m = model(cli)

    generation = m.generate(PROMPT)

    args, kwargs, stdin, files = cli.seen[0]
    assert args[0] == cli.resolved
    assert args[1] == "-p"
    flag = lambda name: args[args.index(name) + 1]
    assert (flag("--model"), flag("--effort")) == ("claude-sonnet-5-5", "medium")
    assert flag("--tools") == ""
    assert flag("--setting-sources") == ""
    assert flag("--output-format") == "json"
    for stripping in ("--disable-slash-commands", "--strict-mcp-config", "--no-session-persistence"):
        assert stripping in args
    assert "--bare" not in args
    # The system prompt is a file in the call's own folder; the user turn is stdin.
    assert flag("--system-prompt-file") == os.path.join(kwargs["cwd"], "system.md")
    assert files == {"system.md": "Extract claims."}
    assert stdin == "supports healthy joints."
    # Outside the repo, deleted afterwards, in its own process group, no API key.
    assert not kwargs["cwd"].startswith(str(cm.config.REPO_ROOT))
    assert not os.path.exists(kwargs["cwd"])
    assert kwargs["start_new_session"] is True
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]

    assert generation.text == "P: supports healthy joints\nEND"
    assert generation.stop_reason == "end_turn"
    # The pinned model's counts only: the Haiku side call is not this answer's.
    assert generation.usage == Usage(prompt_tokens=2 + 100 + 1619, completion_tokens=24)
    assert generation.model == "claude-sonnet-5-5@claude-code-2.1.310"
    assert generation.latency_ms >= 0
    m.close()


def test_an_answer_from_another_model_is_a_failed_request(cli):
    cli.calls = [(answer(modelUsage={"claude-haiku-5-5": {"inputTokens": 1, "outputTokens": 1}}), 0)]

    with pytest.raises(RuntimeError, match="claude-sonnet-5-5"):
        model(cli).generate(PROMPT)


# --- failures ---------------------------------------------------------------


@pytest.mark.parametrize(
    "step, says",
    [
        ((answer(), 1), "exited with code 1"),
        ((answer(is_error=True, subtype="error_during_execution", result="boom"), 0), "boom"),
        (("not json", 0), "not json"),
    ],
)
def test_a_failed_call_is_the_requests_own_error(cli, step, says):
    cli.calls = [step]
    m = model(cli)

    with pytest.raises(RuntimeError, match=says) as raised:
        m.generate(PROMPT)

    assert not isinstance(raised.value, ModelUnavailable)
    m.close()


def test_a_call_past_its_timeout_has_its_process_group_killed(cli):
    cli.calls = ["timeout"]
    m = model(cli, timeout_s=5)

    with pytest.raises(RuntimeError, match="timed out after 5s"):
        m.generate(PROMPT)

    assert cli.killed == [4242]
    m.close()


# --- rate limits and stopping -----------------------------------------------

RATE_LIMITED = (answer(is_error=True, api_error_status=429, result="API Error: 429 rate_limit_error"), 1)


def test_a_rate_limit_is_waited_out_doubling_and_the_call_re_run(cli):
    cli.calls = [RATE_LIMITED, RATE_LIMITED, (answer(), 0)]
    m = model(cli)

    generation = m.generate(PROMPT)

    assert generation.text.endswith("END")
    assert cli.slept == [60, 120]
    assert len(cli.seen) == 3
    m.close()


def test_a_usage_limit_message_counts_as_a_rate_limit(cli):
    cli.calls = [(answer(is_error=True, result="You've hit your limit · resets 3pm"), 1), (answer(), 0)]
    m = model(cli)

    m.generate(PROMPT)

    assert cli.slept == [60]
    m.close()


def test_a_rate_limit_past_the_last_wait_stops_the_model(cli):
    cli.calls = [RATE_LIMITED] * 4
    m = model(cli)

    with pytest.raises(ModelUnavailable, match="rate limit"):
        m.generate(PROMPT)

    assert cli.slept == [60, 120, 240]
    m.close()


def test_a_stopped_model_stays_stopped_until_closed(cli):
    cli.calls = [RATE_LIMITED] * 4 + [(answer(), 0)]
    m = model(cli)
    with pytest.raises(ModelUnavailable):
        m.generate(PROMPT)

    # Nothing more is started: every verb says so at once.
    for verb in (m.ready, lambda: m.generate(PROMPT), lambda: m.submit(PROMPT)):
        with pytest.raises(ModelUnavailable, match="rate limit"):
            verb()
    assert len(cli.seen) == 4

    m.close()
    assert m.generate(PROMPT).text.endswith("END")
    m.close()


def test_failed_calls_in_a_row_stop_the_model(cli):
    cli.calls = [(answer(), 1)] * 3
    m = model(cli, max_consecutive_failures=3)

    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            m.generate(PROMPT)
        assert not isinstance(raised.value, ModelUnavailable)
    with pytest.raises(ModelUnavailable, match="3 calls in a row"):
        m.generate(PROMPT)
    m.close()


def test_a_success_resets_the_count_even_after_waiting_out_a_rate_limit(cli):
    cli.calls = [(answer(), 1), RATE_LIMITED, (answer(), 0), (answer(), 1), (answer(), 1)]
    m = model(cli, max_consecutive_failures=3)

    with pytest.raises(RuntimeError):
        m.generate(PROMPT)
    m.generate(PROMPT)
    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            m.generate(PROMPT)
        assert not isinstance(raised.value, ModelUnavailable)
    m.close()


# --- submit and close -------------------------------------------------------


def test_submit_runs_at_most_n_parallel_calls_and_close_kills_them(cli):
    cli.calls = ["block"] * 3
    m = model(cli, n_parallel=2)

    futures = [m.submit(PROMPT) for _ in range(3)]
    deadline = time.monotonic() + 5
    while cli.running < 2 and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert cli.running == 2

    m.close()

    # Both running calls killed with their groups; the queued one never ran.
    assert sorted(cli.killed) == [4242, 4243]
    assert futures[2].cancelled()
    for f in futures[:2]:
        with pytest.raises(RuntimeError):
            f.result(timeout=5)
    assert cli.most_running == 2
    assert len(cli.seen) == 2


def test_close_is_idempotent_and_a_closed_model_can_be_readied_again(cli):
    m = model(cli)
    m.close()
    m.ready()
    m.close()
    m.close()
    m.ready()
    m.close()


def test_a_call_waiting_out_a_rate_limit_starts_nothing_once_closed(cli, monkeypatch):
    cli.calls = [RATE_LIMITED, (answer(), 0)]
    m = model(cli)
    monkeypatch.setattr(time, "sleep", lambda s: m.close())  # closed during the wait

    with pytest.raises(cm.CancelledError):
        m.generate(PROMPT)

    assert len(cli.seen) == 1
