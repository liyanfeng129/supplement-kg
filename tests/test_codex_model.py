"""CodexModel: OpenAI's `codex exec` on the ChatGPT login, one call per prompt.

Pinned here: settings come from `configs/codex.toml` with explicit arguments
winning; `ready()` resolves the binary once and refuses one that is missing
or not logged in, naming the fix; `generate()` runs one stripped, isolated
`codex exec` call, reads the answer from its `-o` file and the usage from its
`--json` events; a tool event or an `error` item fails the call; a rate limit
is waited out and, past the last wait, stops the model; failed calls are the
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

from engine.models import codex_model as xm
from engine.models.base_model import ModelUnavailable, Prompt, Usage
from engine.models.codex_model import CodexModel

PROMPT = Prompt(system="Extract claims.", user="supports healthy joints.")
TEXT = "P: supports healthy joints\nEND"


def events(*items: dict, usage=True) -> str:
    """The `--json` event lines of one turn, as measured on 0.162 and 0.164."""
    lines = [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"}]
    lines += list(items)
    if usage:
        lines.append(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10591, "cached_input_tokens": 8064, "output_tokens": 19},
            }
        )
    return "\n".join(json.dumps(line) for line in lines) + "\n"


ANSWERED = (events({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": TEXT}}), 0, TEXT)


def failed_turn(message: str) -> tuple:
    return (
        events(
            {"type": "error", "message": message},
            {"type": "turn.failed", "error": {"message": message}},
            usage=False,
        ),
        1,
        None,
    )


RATE_LIMITED = failed_turn('{"type":"error","status":429,"error":{"message":"rate limit reached"}}')


class FakeCli:
    """Stands in for the CLI: `subprocess.run` for checks, `Popen` for calls.

    `calls` is a script: each entry is `(stdout, returncode, last message)`
    -- the message is written to the `-o` file unless None -- `"timeout"`
    for a call that never finishes, or `"block"` for one that runs until its
    process group is killed. What each call was given is kept in `seen` as
    `(args, kwargs, stdin, files in its cwd)`.
    """

    def __init__(self, calls=(), version="codex-cli 0.164.0-alpha.11.1", logged_in=True, version_rc=0):
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
        if args[1:] == ["login", "status"]:
            # The status line goes to stderr, after a harmless warning.
            said = "Logged in using ChatGPT" if self.logged_in else "Not logged in"
            return subprocess.CompletedProcess(
                args, 0 if self.logged_in else 1, "", f"WARNING: stale arg0 temp dirs\n{said}\n"
            )
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
                stdout, self.returncode, last = step
                if last is not None:
                    with open(args[args.index("-o") + 1], "w") as f:
                        f.write(last)
                return stdout, "stderr text"

        return Proc()

    def killpg(self, pid, sig):
        assert sig == signal.SIGKILL
        self.killed.append(pid)
        self.released.set()


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """A fake CLI, and a real binary path: a file inside an extension folder."""
    binary = tmp_path / "openai.chatgpt-26.1021.22027-linux-x64" / "codex"
    binary.parent.mkdir()
    binary.write_text("")
    fake = FakeCli()
    monkeypatch.setattr(subprocess, "run", fake.run)
    monkeypatch.setattr(subprocess, "Popen", fake.popen)
    monkeypatch.setattr(os, "killpg", fake.killpg)
    monkeypatch.setattr(time, "sleep", fake.slept.append)
    fake.binary = str(binary)
    return fake


def model(cli, **kwargs) -> CodexModel:
    return CodexModel(binary=cli.binary, **kwargs)


# --- settings ---------------------------------------------------------------


def test_the_committed_file_supplies_the_settings():
    m = CodexModel()

    assert m.binary.startswith(os.path.expanduser("~/.vscode-server/extensions/openai.chatgpt-"))
    assert m.binary.endswith("/bin/linux-x86_64/codex")
    assert (m.model, m.effort) == ("gpt-6-luna", "medium")
    assert (m.n_parallel, m.timeout_s) == (4, 300)
    assert (m.rate_limit_wait_s, m.rate_limit_waits, m.max_consecutive_failures) == (60, 3, 8)


def test_an_explicit_argument_beats_the_file():
    assert CodexModel(effort="high").effort == "high"


def test_a_missing_file_falls_back_to_the_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(xm.config, "CODEX_CONFIG_PATH", tmp_path / "missing.toml")

    m = CodexModel()

    assert (m.binary, m.model, m.n_parallel) == ("", "gpt-6-luna", 4)


# --- ready ------------------------------------------------------------------


def test_ready_checks_the_binary_once(cli):
    m = model(cli)

    m.ready()
    m.ready()

    assert cli.run_args == [[cli.binary, "--version"], [cli.binary, "login", "status"]]
    m.close()


def test_ready_refuses_a_binary_that_moved_naming_the_toml_line(cli, tmp_path):
    m = CodexModel(binary=str(tmp_path / "openai.chatgpt-old" / "codex"))

    with pytest.raises(ModelUnavailable, match=r"\[model\]\.binary"):
        m.ready()


def test_ready_refuses_a_binary_that_does_not_run(cli):
    cli.version_rc = 127

    with pytest.raises(ModelUnavailable, match="--version"):
        model(cli).ready()


def test_ready_refuses_a_cli_that_is_not_logged_in(cli):
    cli.logged_in = False

    with pytest.raises(ModelUnavailable, match="codex login"):
        model(cli).ready()


# --- generate ---------------------------------------------------------------


def test_one_call_is_stripped_isolated_and_read_into_a_generation(cli, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-reach-the-cli")
    monkeypatch.setenv("CODEX_API_KEY", "sk-should-not-reach-the-cli")
    cli.calls = [ANSWERED]
    m = model(cli)

    generation = m.generate(PROMPT)

    args, kwargs, stdin, files = cli.seen[0]
    temp = kwargs["cwd"]
    assert args[:2] == [cli.binary, "exec"]
    flag = lambda name: args[args.index(name) + 1]
    assert flag("-m") == "gpt-6-luna"
    configs = [args[i + 1] for i, a in enumerate(args) if a == "-c"]
    assert configs == [
        'model_reasoning_effort="medium"',
        f"model_instructions_file='{os.path.join(temp, 'instructions.md')}'",
    ]
    assert (flag("-C"), flag("-s")) == (temp, "read-only")
    for stripping in ("--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "--json"):
        assert stripping in args
    assert flag("-o") == os.path.join(temp, "last.txt")
    assert args[-1] == "-"
    # The system prompt is a file in the call's own folder; the user turn is stdin.
    assert files == {"instructions.md": "Extract claims."}
    assert stdin == "supports healthy joints."
    # Outside the repo, deleted afterwards, in its own process group, no API key.
    assert not temp.startswith(str(xm.config.REPO_ROOT))
    assert not os.path.exists(temp)
    assert kwargs["start_new_session"] is True
    assert "OPENAI_API_KEY" not in kwargs["env"] and "CODEX_API_KEY" not in kwargs["env"]

    assert generation.text == TEXT
    assert generation.stop_reason == ""
    assert generation.usage == Usage(prompt_tokens=10591, completion_tokens=19)
    assert generation.model == "gpt-6-luna@codex-cli-0.164.0-alpha.11.1"
    m.close()


# --- failures ---------------------------------------------------------------


@pytest.mark.parametrize(
    "step, says",
    [
        ((events(), 1, TEXT), "exited with code 1"),
        (failed_turn("model not supported"), "model not supported"),
        # A setting Codex ignores shows only as an error item, and the prompt
        # sent is then not the prompt asked for.
        (
            (
                events(
                    {"type": "item.completed", "item": {"id": "i", "type": "error", "message": "`base_instructions` is ignored"}},
                    {"type": "item.completed", "item": {"id": "j", "type": "agent_message", "text": TEXT}},
                ),
                0,
                TEXT,
            ),
            "base_instructions",
        ),
    ],
)
def test_a_failed_call_is_the_requests_own_error(cli, step, says):
    cli.calls = [step]
    m = model(cli)

    with pytest.raises(RuntimeError, match=says) as raised:
        m.generate(PROMPT)

    assert not isinstance(raised.value, ModelUnavailable)
    m.close()


@pytest.mark.parametrize("kind", ["command_execution", "file_change", "mcp_tool_call", "web_search"])
def test_a_tool_event_discards_the_answer(cli, kind):
    cli.calls = [
        (
            events(
                {"type": "item.started", "item": {"id": "i", "type": kind}},
                {"type": "item.completed", "item": {"id": "j", "type": "agent_message", "text": TEXT}},
            ),
            0,
            TEXT,
        )
    ]
    m = model(cli)

    with pytest.raises(RuntimeError, match=kind):
        m.generate(PROMPT)
    m.close()


def test_a_call_past_its_timeout_has_its_process_group_killed(cli):
    cli.calls = ["timeout"]
    m = model(cli, timeout_s=5)

    with pytest.raises(RuntimeError, match="timed out after 5s"):
        m.generate(PROMPT)

    assert cli.killed == [4242]
    m.close()


# --- rate limits and stopping -----------------------------------------------


def test_a_rate_limit_is_waited_out_doubling_and_the_call_re_run(cli):
    cli.calls = [RATE_LIMITED, RATE_LIMITED, ANSWERED]
    m = model(cli)

    assert m.generate(PROMPT).text == TEXT
    assert cli.slept == [60, 120]
    m.close()


def test_a_rate_limit_past_the_last_wait_stops_the_model_until_closed(cli):
    cli.calls = [RATE_LIMITED] * 4 + [ANSWERED]
    m = model(cli)

    with pytest.raises(ModelUnavailable, match="rate limit"):
        m.generate(PROMPT)
    assert cli.slept == [60, 120, 240]
    for verb in (m.ready, lambda: m.generate(PROMPT), lambda: m.submit(PROMPT)):
        with pytest.raises(ModelUnavailable, match="rate limit"):
            verb()
    assert len(cli.seen) == 4

    m.close()
    assert m.generate(PROMPT).text == TEXT
    m.close()


def test_failed_calls_in_a_row_stop_the_model_and_a_success_resets_the_count(cli):
    bad = failed_turn("boom")
    cli.calls = [bad, bad, ANSWERED, bad, bad, bad]
    m = model(cli, max_consecutive_failures=3)

    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            m.generate(PROMPT)
        assert not isinstance(raised.value, ModelUnavailable)
    m.generate(PROMPT)
    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            m.generate(PROMPT)
        assert not isinstance(raised.value, ModelUnavailable)
    with pytest.raises(ModelUnavailable, match="3 calls in a row"):
        m.generate(PROMPT)
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

    assert sorted(cli.killed) == [4242, 4243]
    assert futures[2].cancelled()
    for f in futures[:2]:
        with pytest.raises(RuntimeError):
            f.result(timeout=5)
    assert cli.most_running == 2
    assert len(cli.seen) == 2


def test_a_call_waiting_out_a_rate_limit_starts_nothing_once_closed(cli, monkeypatch):
    cli.calls = [RATE_LIMITED, ANSWERED]
    m = model(cli)
    monkeypatch.setattr(time, "sleep", lambda s: m.close())  # closed during the wait

    with pytest.raises(xm.CancelledError):
        m.generate(PROMPT)

    assert len(cli.seen) == 1


def test_a_turn_without_a_last_message_is_a_failed_request(cli):
    cli.calls = [(events(), 0, None)]
    m = model(cli)

    with pytest.raises(RuntimeError, match="no -o file"):
        m.generate(PROMPT)
    m.close()
