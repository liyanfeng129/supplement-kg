"""CodexModel: the Responses API on the ChatGPT plan, one streamed call per prompt.

Pinned here: settings come from `configs/codex.toml` with explicit arguments
winning; `ready()` loads the signed-in credentials and makes one test call,
once, refusing missing credentials or a refused call by asking to sign in
again; `generate()` sends the prompt as instructions plus one user message and
no tools, and reads the text and usage from the event stream; a token about to
expire is refreshed first and stored `0600`, and a refused refresh stops the
model; a refusal no retry fixes stops the model; a tool item fails the call; a
rate limit or the plan's usage limit is waited out and, past the last wait,
stops the model; failed calls are the request's own error until
`max_consecutive_failures` in a row stop the model; a stopped model stays
stopped until `close()`; `submit()` runs at most `n_parallel` at once and
`close()` cuts running streams.

Only public functions are called. No request leaves: underneath the model,
`urllib.request.urlopen` and `time.sleep` are replaced. `sign_in()` is
interactive and checked live in notebook 15.
"""

from __future__ import annotations

import io
import json
import os
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from engine.models import codex_model as xm
from engine.models.base_model import ModelUnavailable, Prompt, Usage
from engine.models.codex_model import CodexModel

PROMPT = Prompt(system="Extract claims.", user="supports healthy joints.")
TEXT = "P: supports healthy joints\nEND"
CLIENT = "oaiapp_test"


def stream(*events: dict) -> list[bytes]:
    """The server-sent event lines of one streamed response."""
    return [f"data: {json.dumps(e)}\n".encode() for e in events] + [b"data: [DONE]\n"]


def answered(text: str = TEXT, model: str = "gpt-6.1-terra") -> tuple:
    return ("stream", stream(
        {"type": "response.created", "response": {"status": "in_progress"}},
        {"type": "response.output_item.added", "item": {"type": "message"}},
        *({"type": "response.output_text.delta", "delta": part} for part in (text[:5], text[5:])),
        {"type": "response.completed", "response": {
            "status": "completed", "model": model,
            "usage": {"input_tokens": 2180, "output_tokens": 83}}},
    ))


def failed(code: str) -> tuple:
    return ("stream", stream({"type": "response.failed", "response": {"status": "failed", "error": {"code": code}}}))


def http(status: int, code: str | None = None) -> tuple:
    return ("http", status, {"error": {"code": code}} if code else {})


OK = answered("OK")  # what ready()'s test call gets
USAGE_LIMIT = failed("subscription_sharing_usage_limit_exceeded")


class FakeOpenAI:
    """Stands in for `urlopen`: the Responses endpoint and the token endpoint.

    `calls` scripts the Responses calls in order: `("stream", lines)`,
    `("http", status, body)`, or `"block"` for a stream that runs until
    closed. `refreshes` scripts the token endpoint the same way, with
    `("token", body)` for a success. Every request is kept in `seen` as
    `(url, headers, body)`.
    """

    def __init__(self):
        self.calls: list = []
        self.refreshes: list = []
        self.seen: list[tuple] = []
        self.slept: list[float] = []
        self.lock = threading.Lock()
        self.running = 0
        self.most_running = 0

    def urlopen(self, req, timeout=None):
        url = req.full_url
        headers = {k.lower(): v for k, v in req.header_items()}
        with self.lock:
            self.seen.append((url, headers, req.data))
            step = (self.refreshes if url == xm._TOKEN_URL else self.calls).pop(0)
        if step == "block":
            return Blocking(self)
        if step[0] == "http":
            raise urllib.error.HTTPError(url, step[1], "error", {}, io.BytesIO(json.dumps(step[2]).encode()))
        if step[0] == "token":
            return io.BytesIO(json.dumps(step[1]).encode())
        return io.BytesIO(b"".join(step[1]))

    def responses(self) -> list[dict]:
        return [json.loads(body) for url, _, body in self.seen if url == xm._RESPONSES_URL]


class Blocking:
    """A stream that yields nothing until `close()` cuts it."""

    def __init__(self, api):
        self.api = api
        self.closed = threading.Event()

    def __iter__(self):
        with self.api.lock:
            self.api.running += 1
            self.api.most_running = max(self.api.most_running, self.api.running)
        self.closed.wait(5)
        with self.api.lock:
            self.api.running -= 1
        raise ValueError("I/O operation on closed file")

    def close(self):
        self.closed.set()


@pytest.fixture
def api(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(time, "sleep", fake.slept.append)
    return fake


@pytest.fixture
def home(tmp_path):
    """Credentials as sign_in() leaves them, valid for an hour."""
    d = tmp_path / "chatgpt-plan"
    d.mkdir()
    (d / "current.json").write_text(json.dumps({"client_id": CLIENT}))
    (d / f"{CLIENT}.json").write_text(json.dumps({
        "client_id": CLIENT, "sub": "user-1", "access_token": "at-1", "refresh_token": "rt-1",
        "id_token": "x.y.z", "scope": "openid offline_access resource.invoke chatgpt.tokens.use.direct",
        "expires_at": time.time() + 3600,
    }))
    return d


def model(home, **kwargs) -> CodexModel:
    return CodexModel(credentials_dir=str(home), **kwargs)


# --- settings ---------------------------------------------------------------


def test_the_committed_file_supplies_the_settings():
    m = CodexModel()

    assert m.credentials_dir == os.path.expanduser("~/.config/supplement-kg/chatgpt-plan")
    assert (m.model, m.effort) == ("gpt-6.1-terra", "medium")
    assert (m.n_parallel, m.timeout_s) == (8, 300)
    assert (m.rate_limit_wait_s, m.rate_limit_waits, m.max_consecutive_failures) == (60, 3, 16)


def test_an_explicit_argument_beats_the_file():
    assert CodexModel(effort="high").effort == "high"


def test_a_missing_file_falls_back_to_the_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(xm.config, "CODEX_CONFIG_PATH", tmp_path / "missing.toml")

    m = CodexModel()

    assert (m.credentials_dir, m.model, m.n_parallel) == (
        os.path.expanduser("~/.config/supplement-kg/chatgpt-plan"), "gpt-6.1-terra", 4)


# --- ready ------------------------------------------------------------------


def test_ready_makes_one_test_call_once(api, home):
    api.calls = [OK]
    m = model(home)

    m.ready()
    m.ready()

    assert len(api.responses()) == 1
    assert api.responses()[0]["instructions"] == "Reply with OK."
    m.close()


def test_ready_without_a_sign_in_asks_for_one(api, tmp_path):
    with pytest.raises(ModelUnavailable, match=r"sign_in\(\)"):
        model(tmp_path / "nowhere").ready()
    assert api.seen == []


def test_ready_refuses_a_sign_in_without_the_plan_scope(api, home):
    cred = json.loads((home / f"{CLIENT}.json").read_text())
    (home / f"{CLIENT}.json").write_text(json.dumps({**cred, "scope": "openid email"}))

    with pytest.raises(ModelUnavailable, match="chatgpt.tokens.use.direct"):
        model(home).ready()
    assert api.seen == []


@pytest.mark.parametrize("refused", [http(401), http(401, "subscription_sharing_invalid_user")])
def test_a_test_call_refused_asks_to_sign_in_again(api, home, refused):
    api.calls = [refused]
    m = model(home)

    with pytest.raises(ModelUnavailable, match=r"expired or was revoked.*sign_in\(\)"):
        m.ready()
    with pytest.raises(ModelUnavailable, match="sign_in"):
        m.submit(PROMPT)
    assert len(api.responses()) == 1


def test_a_test_call_that_fails_otherwise_stops_naming_the_error(api, home):
    api.calls = [http(503, "subscription_sharing_usage_unavailable")]

    with pytest.raises(ModelUnavailable, match="test call failed.*503"):
        model(home).ready()


# --- generate ---------------------------------------------------------------


def test_one_call_sends_the_prompt_alone_and_is_read_into_a_generation(api, home):
    api.calls = [OK, answered()]
    m = model(home)

    generation = m.generate(PROMPT)

    url, headers, body = api.seen[-1]
    assert url == "https://api.openai.com/v1/responses"
    assert headers["authorization"] == "Bearer at-1"
    assert json.loads(body) == {
        "model": "gpt-6.1-terra",
        "instructions": "Extract claims.",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "supports healthy joints."}]}],
        "reasoning": {"effort": "medium"},
        "store": False,
        "stream": True,
    }
    assert generation.text == TEXT
    assert generation.usage == Usage(prompt_tokens=2180, completion_tokens=83)
    assert generation.stop_reason == "completed"
    assert generation.model == "gpt-6.1-terra@responses"
    m.close()


def test_a_token_about_to_expire_is_refreshed_and_stored_private(api, home):
    cred = json.loads((home / f"{CLIENT}.json").read_text())
    (home / f"{CLIENT}.json").write_text(json.dumps({**cred, "expires_at": time.time() + 60}))
    api.refreshes = [("token", {"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600,
                                "scope": cred["scope"]})]
    api.calls = [OK]

    model(home).ready()

    url, _, body = api.seen[0]
    assert url == xm._TOKEN_URL
    assert urllib.parse.parse_qs(body.decode()) == {
        "grant_type": ["refresh_token"], "client_id": [CLIENT], "refresh_token": ["rt-1"],
        "resource": ["https://api.openai.com/v1"]}
    assert api.seen[1][1]["authorization"] == "Bearer at-2"
    stored = home / f"{CLIENT}.json"
    assert json.loads(stored.read_text())["refresh_token"] == "rt-2"
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600


def test_a_refused_refresh_asks_to_sign_in_again(api, home):
    cred = json.loads((home / f"{CLIENT}.json").read_text())
    (home / f"{CLIENT}.json").write_text(json.dumps({**cred, "expires_at": time.time() - 10}))
    api.refreshes = [("http", 400, {"error": "invalid_grant"})]

    with pytest.raises(ModelUnavailable, match=r"invalid_grant.*sign_in\(\)"):
        model(home).ready()
    assert api.responses() == []


# --- failures ---------------------------------------------------------------


@pytest.mark.parametrize(
    "step, says",
    [
        (http(503, "subscription_sharing_usage_unavailable"), "503"),
        (http(500), "500"),
        (failed("server_error"), "server_error"),
        (("stream", stream({"type": "response.incomplete",
                            "response": {"incomplete_details": {"reason": "max_output_tokens"}}})), "max_output_tokens"),
        (("stream", [b"data: {\"type\": \"response.created\"}\n"]), "without a final event"),
    ],
)
def test_a_failed_call_is_the_requests_own_error(api, home, step, says):
    api.calls = [OK, step]
    m = model(home)
    m.ready()

    with pytest.raises(RuntimeError, match=says) as raised:
        m.generate(PROMPT)

    assert not isinstance(raised.value, ModelUnavailable)
    m.close()


@pytest.mark.parametrize("code", ["subscription_sharing_user_not_eligible",
                                  "subscription_sharing_unsupported_capability"])
def test_a_refusal_no_retry_fixes_stops_the_model(api, home, code):
    api.calls = [OK, http(403 if "eligible" in code else 400, code)]
    m = model(home)
    m.ready()

    with pytest.raises(ModelUnavailable, match=code):
        m.generate(PROMPT)
    with pytest.raises(ModelUnavailable, match=code):
        m.submit(PROMPT)
    m.close()


def test_a_tool_item_discards_the_answer(api, home):
    api.calls = [OK, ("stream", stream(
        {"type": "response.output_item.added", "item": {"type": "function_call"}},
        {"type": "response.completed", "response": {"status": "completed"}},
    ))]
    m = model(home)
    m.ready()

    with pytest.raises(RuntimeError, match="function_call"):
        m.generate(PROMPT)
    m.close()


# --- rate limits and stopping -----------------------------------------------


def test_a_rate_limit_is_waited_out_doubling_and_the_call_re_run(api, home):
    api.calls = [OK, http(429), USAGE_LIMIT, answered()]
    m = model(home)
    m.ready()

    assert m.generate(PROMPT).text == TEXT
    assert api.slept == [60, 120]
    m.close()


def test_a_usage_limit_past_the_last_wait_stops_the_model_until_closed(api, home):
    api.calls = [OK] + [USAGE_LIMIT] * 4 + [OK, answered()]
    m = model(home)
    m.ready()

    with pytest.raises(ModelUnavailable, match="usage_limit_exceeded"):
        m.generate(PROMPT)
    assert api.slept == [60, 120, 240]
    for verb in (m.ready, lambda: m.generate(PROMPT), lambda: m.submit(PROMPT)):
        with pytest.raises(ModelUnavailable, match="rate limited"):
            verb()
    assert len(api.responses()) == 5

    m.close()
    assert m.generate(PROMPT).text == TEXT  # ready() again: a new test call, then the prompt
    m.close()


def test_failed_calls_in_a_row_stop_the_model_and_a_success_resets_the_count(api, home):
    bad = http(500)
    api.calls = [OK, bad, bad, answered(), bad, bad, bad]
    m = model(home, max_consecutive_failures=3)
    m.ready()

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


def test_submit_runs_at_most_n_parallel_calls_and_close_cuts_them(api, home):
    api.calls = [OK] + ["block"] * 3
    m = model(home, n_parallel=2)
    m.ready()

    futures = [m.submit(PROMPT) for _ in range(3)]
    deadline = time.monotonic() + 5
    while api.running < 2 and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert api.running == 2

    m.close()

    assert futures[2].cancelled()
    for f in futures[:2]:
        with pytest.raises(xm.CancelledError):
            f.result(timeout=5)
    assert api.most_running == 2
    assert len(api.responses()) == 3  # the test call and the two that ran
