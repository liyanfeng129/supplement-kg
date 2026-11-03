"""Codex: OpenAI's Responses API on the user's ChatGPT plan, never an API key.

**One streamed `POST /v1/responses` per prompt, and that is this file's private
business.** Nothing above knows a transport exists. The request carries an
OAuth token from Sign in with ChatGPT, so it bills the user's ChatGPT plan
(https://developers.openai.com/siwc/token-sharing-open-source). It sends
`Prompt.system` as `instructions`, `Prompt.user` as the one input message, and
no tools: the model sees the prompt and nothing else.

**Signing in is once per machine.** `sign_in()` runs the OAuth flow in the
user's browser and stores the credentials in `credentials_dir`, outside the
repo, `0600`. The access token lasts an hour and is refreshed before a call
that would outlive it; the refresh token lasts 30 days and renews itself.

**`ready()` makes one test call.** A sign-in can expire or be revoked, and only
a completed call shows it. One refused with 401/403, missing credentials or a
refused refresh raise `ModelUnavailable` asking to sign in again.

**Not one per process.** No singleton: a call holds nothing exclusive.
`n_parallel` calls run at once, in a pool `ready()` starts.

**Tool guard.** No tool is sent, but an output item other than a message or
reasoning fails the call anyway.

**Not deterministic.** This route accepts no temperature, seed or output cap,
so `Prompt.max_tokens`, `Prompt.stop` and `Prompt.schema` are ignored; a
cut-off answer fails the prompt's own checks.

**Stopping.** A rate limit or the plan's usage limit is waited out; one that
outlasts the waits raises `ModelUnavailable`, as do `max_consecutive_failures`
failed calls in a row and any refusal no retry can fix. A model that raised it
stays stopped until `close()`.

**Settings come from `configs/codex.toml`**; an explicit argument wins. See
`docs/specs/0012-claude-codex-model-controller.md`, "Codex on the plan token".

Public: `ready`, `generate`, `submit`, `close`, `sign_in`. Everything else is
private.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import http.server
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
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

# Used where `configs/codex.toml` says nothing.
_FALLBACKS: dict[str, Any] = {
    "credentials_dir": "~/.config/supplement-kg/chatgpt-plan",
    "model": "gpt-6.1-terra",
    "effort": "medium",
    "n_parallel": 4,
    "timeout_s": 300,
    "rate_limit_wait_s": 60,
    "rate_limit_waits": 3,
    "max_consecutive_failures": 8,
}

_RESPONSES_URL = "https://api.openai.com/v1/responses"
_AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
_TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
_RESOURCE = "https://api.openai.com/v1"
_REDIRECT = "http://127.0.0.1:1455/auth/callback"
_SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
_PLAN_SCOPE = "chatgpt.tokens.use.direct"
_APP_NAME = "supplement-kg judge"

# A token that expires within this is refreshed before a call.
_REFRESH_MARGIN_S = 300

# What ready()'s test call sends: ~20 tokens.
_TEST_PROMPT = Prompt(system="Reply with OK.", user="OK")

_SIGN_IN_AGAIN = (
    "the ChatGPT sign-in has expired or was revoked; run `CodexModel().sign_in()` and "
    "sign in again"
)

# Error codes after which no retry helps (spec 0012, Codex on the plan token).
_REFUSED = {
    "subscription_sharing_user_not_eligible",
    "subscription_sharing_unsupported_capability",
    "subscription_sharing_route_not_supported",
    "subscription_sharing_invalid_user",
    "chatpass_v2_scope_not_authorized",
    "chatpass_v2_invalid_authorization_context",
}

# Output items that are the model talking, not a tool at work.
_ANSWER_ITEMS = {"message", "reasoning"}


class CodexModel(BaseModel):
    """The Responses API on the ChatGPT plan, `n_parallel` calls at once.

    Build it with `CodexModel(...)` -- free, runs nothing. `ready()` loads the
    credentials and makes one test call; `close()` stops whatever is running.
    `sign_in()` is the one interactive verb.
    """

    # The settings. All come from `configs/codex.toml` unless passed.
    credentials_dir: str  # `~` expanded; outside the repo
    model: str  # the pinned model id
    effort: str
    n_parallel: int  # calls running at once
    timeout_s: int  # per call
    rate_limit_wait_s: int  # the first wait; each next one doubles
    rate_limit_waits: int
    max_consecutive_failures: int

    def __init__(
        self,
        credentials_dir: str | None = None,
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
            "credentials_dir": credentials_dir,
            "model": model,
            "effort": effort,
            "n_parallel": n_parallel,
            "timeout_s": timeout_s,
            "rate_limit_wait_s": rate_limit_wait_s,
            "rate_limit_waits": rate_limit_waits,
            "max_consecutive_failures": max_consecutive_failures,
        }
        settings = dict(_FALLBACKS)
        section = config.read_toml(config.CODEX_CONFIG_PATH).get("model")
        if isinstance(section, dict):
            # Only known keys: a typo in the file is ignored, not fatal.
            settings.update({k: v for k, v in section.items() if k in _FALLBACKS})
        settings.update({k: v for k, v in asked.items() if v is not None})
        settings["credentials_dir"] = os.path.expanduser(settings["credentials_dir"])
        for key, value in settings.items():
            setattr(self, key, value)

        self._cred: dict | None = None  # the signed-in client's tokens, loaded by ready()
        self._pool: ThreadPoolExecutor | None = None
        self._running: set[Any] = set()  # open response streams, for close() to cut
        self._failures = 0  # failed calls in a row
        self._stopped = ""  # why the model raised ModelUnavailable; "" while fit
        self._lock = threading.RLock()

    # --- lifecycle ----------------------------------------------------------

    def ready(self) -> None:
        """Load the credentials, make one test call, start the pool. Idempotent.

        The test call is ~20 tokens, once per `ready()` that starts a pool: a
        listed model is a catalog entry, not an entitlement, and only a
        completed call shows the plan still accepts this sign-in. Any failure
        raises `ModelUnavailable` naming the fix.
        """
        with self._lock:
            if self._stopped:
                raise ModelUnavailable(self._stopped)
            if self._pool is not None:
                return

            # 1. The credentials sign_in() stored, with the plan scope granted.
            home = Path(self.credentials_dir)
            try:
                client_id = json.loads((home / "current.json").read_text())["client_id"]
                cred = json.loads((home / f"{client_id}.json").read_text())
            except (OSError, ValueError, KeyError):
                raise ModelUnavailable(
                    f"Refusing to run: no ChatGPT sign-in in {home}. Run `CodexModel().sign_in()`."
                ) from None
            if _PLAN_SCOPE not in str(cred.get("scope", "")).split():
                raise ModelUnavailable(
                    f"Refusing to run: the sign-in lacks the {_PLAN_SCOPE} scope; {_SIGN_IN_AGAIN}."
                )
            self._cred = cred

            # 2. The pool, then the test call through generate(), which refreshes
            # the token if it must. A failure takes the pool down again.
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, self.n_parallel), thread_name_prefix="codex-call"
            )
            try:
                self.generate(_TEST_PROMPT)
            except ModelUnavailable:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None
                raise
            except Exception as exc:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None
                raise ModelUnavailable(
                    f"Refusing to run: the test call failed: {exc}. If it keeps failing, the "
                    "sign-in has probably expired: run `CodexModel().sign_in()`."
                ) from exc

    def close(self) -> None:
        """Cancel queued prompts, cut running streams, clear a stop. Idempotent.

        A cut call's answer is discarded. `ready()` after `close()` makes the
        model usable again.
        """
        with self._lock:
            pool, self._pool = self._pool, None
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            for resp in list(self._running):
                try:
                    resp.close()
                except Exception:
                    pass  # already closed
            self._failures = 0
            self._stopped = ""

    def sign_in(self) -> None:
        """Sign in with ChatGPT in the browser and store the credentials. Interactive.

        Prints the authorize URL and waits up to 10 minutes for the callback on
        127.0.0.1:1455 (forward that port when the kernel runs over SSH). The
        first sign-in registers this app and gets its own client id; a later
        one reuses it. Never prints a token.
        """
        home = Path(self.credentials_dir)
        home.mkdir(parents=True, exist_ok=True, mode=0o700)

        def write_private(path: Path, data: dict) -> None:
            tmp = path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)

        # 1. The host id: opaque, kept, the same for this machine every time.
        host_file = home / "host.json"
        if host_file.exists():
            host_id = json.loads(host_file.read_text())["ext_agent_host_id"]
        else:
            host_id = f"urn:uuid:{uuid.uuid4()}"
            write_private(host_file, {"ext_agent_host_id": host_id})

        # 2. A returning sign-in reuses the issued client id; a first one registers.
        known: dict = {}
        try:
            known_id = json.loads((home / "current.json").read_text())["client_id"]
            known = json.loads((home / f"{known_id}.json").read_text())
        except (OSError, ValueError, KeyError):
            pass
        state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        params = {
            "client_id": known.get("client_id") or "dynamic_agent_client",
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": _REDIRECT,
            "scope": _SCOPES,
            "resource": _RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if known.get("id_token"):
            params["id_token_hint"] = known["id_token"]
        else:
            params["agent_name_hint"] = _APP_NAME

        # 3. The listener, threaded: an idle connection (VSCode's port forwarder
        # opens one) must not block the callback behind it.
        result: dict = {}

        class Callback(http.server.BaseHTTPRequestHandler):
            timeout = 10

            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                if url.path != "/auth/callback":
                    self.send_response(404)
                    self.end_headers()
                    return
                result.update({k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()})
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Signed in. You can close this tab.")

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 1455), Callback)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Open this URL in your browser and approve:\n\n{_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}\n",
              flush=True)
        try:
            deadline = time.monotonic() + 600
            while "state" not in result and "error" not in result and time.monotonic() < deadline:
                time.sleep(0.5)
        finally:
            server.shutdown()
            server.server_close()

        # 4. The callback: refused, forged or missing stops here.
        if result.get("error"):
            raise ModelUnavailable(f"Sign-in refused: {result['error']} {result.get('error_description', '')}")
        if not result:
            raise ModelUnavailable("Sign-in timed out after 10 minutes; nothing stored.")
        if result.get("state") != state:
            raise ModelUnavailable("Sign-in callback state does not match; nothing stored.")
        if not result.get("code"):
            raise ModelUnavailable("Sign-in callback has no code; nothing stored.")
        client_id = result.get("client_id") or known.get("client_id")
        if not client_id:
            raise ModelUnavailable("Sign-in callback has no issued client id; nothing stored.")

        # 5. The code for tokens.
        body = urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": result["code"],
            "code_verifier": verifier,
            "redirect_uri": _REDIRECT,
            "resource": _RESOURCE,
        }).encode()
        req = urllib.request.Request(_TOKEN_URL, data=body, method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                tok = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise ModelUnavailable(f"Sign-in token exchange failed: HTTP {exc.code}") from None

        # 6. The ID token's claims. It came straight from the token endpoint over
        # TLS, which OIDC Core 3.1.3.7 lets stand in for the signature check.
        payload = tok["id_token"].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        aud = claims.get("aud") if isinstance(claims.get("aud"), list) else [claims.get("aud")]
        problems = [name for name, ok in (
            ("iss", claims.get("iss") == "https://auth.openai.com"),
            ("aud", client_id in aud),
            ("exp", claims.get("exp", 0) > time.time()),
            ("nonce", claims.get("nonce") == nonce),
            ("scope", _PLAN_SCOPE in str(tok.get("scope", "")).split()),
        ) if not ok]
        if problems:
            raise ModelUnavailable(f"Sign-in checks failed ({', '.join(problems)}); nothing stored.")

        # 7. Stored, then marked current; the next ready() loads it.
        write_private(home / f"{client_id}.json", {
            "client_id": client_id,
            "sub": claims["sub"],
            "access_token": tok["access_token"],
            "refresh_token": tok.get("refresh_token"),
            "id_token": tok["id_token"],
            "scope": tok.get("scope"),
            "expires_at": time.time() + int(tok.get("expires_in", 3600)),
        })
        write_private(home / "current.json", {"client_id": client_id})
        # A model stopped by the old sign-in starts afresh: the next ready()
        # loads these credentials and makes its test call.
        self.close()
        print(f"Signed in: client {client_id}, credentials in {home}.", flush=True)

    # --- generation ---------------------------------------------------------

    def generate(self, prompt: Prompt) -> Generation:
        """Answer one prompt with one streamed Responses call.

        Safe from several threads at once, which is how `submit()`'s workers
        run. No retry of a failed answer here: that is the controller's.

        A failed call raises one of two things. `ModelUnavailable` when the
        model must stop: the sign-in refused, a refusal no retry fixes, a rate
        limit past the last wait, or the `max_consecutive_failures`-th failed
        call in a row. Anything else is that call's own error, and the next
        call may well succeed.
        """
        self.ready()
        session = self._pool  # this ready()'s pool; close() ends it
        try:
            waits = 0
            while True:
                # 1. A token that would expire during the call is refreshed first,
                # one refresher at a time across threads and processes: the
                # refresh token rotates, and a race would spend it twice.
                with self._lock:
                    if self._pool is not session:
                        raise CancelledError()
                    if self._stopped:  # another call stopped the model meanwhile
                        raise ModelUnavailable(self._stopped)
                    cred = self._cred
                    if cred["expires_at"] < time.time() + _REFRESH_MARGIN_S:
                        home = Path(self.credentials_dir)
                        with open(home / "refresh.lock", "a") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX)
                            # Another process may have refreshed meanwhile.
                            path = home / f"{cred['client_id']}.json"
                            cred = json.loads(path.read_text())
                            if cred["expires_at"] < time.time() + _REFRESH_MARGIN_S:
                                body = urllib.parse.urlencode({
                                    "grant_type": "refresh_token",
                                    "client_id": cred["client_id"],
                                    "refresh_token": cred["refresh_token"],
                                    "resource": _RESOURCE,
                                }).encode()
                                req = urllib.request.Request(
                                    _TOKEN_URL, data=body, method="POST",
                                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                                )
                                try:
                                    with urllib.request.urlopen(req, timeout=30) as resp:
                                        tok = json.loads(resp.read())
                                except urllib.error.HTTPError as exc:
                                    try:
                                        code = json.loads(exc.read()).get("error")
                                    except ValueError:
                                        code = None
                                    if exc.code in (400, 401) or code in ("invalid_grant", "invalid_client"):
                                        self._stopped = f"Refusing to run: refresh refused ({code or exc.code}); {_SIGN_IN_AGAIN}."
                                        raise ModelUnavailable(self._stopped) from None
                                    raise RuntimeError(f"token refresh failed: HTTP {exc.code}") from None
                                cred = {
                                    **cred,
                                    "access_token": tok["access_token"],
                                    "refresh_token": tok.get("refresh_token") or cred["refresh_token"],
                                    "scope": tok.get("scope", cred.get("scope")),
                                    "expires_at": time.time() + int(tok.get("expires_in", 3600)),
                                }
                                tmp = path.with_suffix(".tmp")
                                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                                with os.fdopen(fd, "w") as fh:
                                    json.dump(cred, fh)
                                os.replace(tmp, path)
                        self._cred = cred
                    token = cred["access_token"]

                # 2. The request: instructions, one user message, no tools.
                body = json.dumps({
                    "model": self.model,
                    "instructions": prompt.system,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt.user}]}],
                    "reasoning": {"effort": self.effort},
                    "store": False,
                    "stream": True,
                }).encode()
                req = urllib.request.Request(
                    _RESPONSES_URL, data=body, method="POST",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                             "Accept": "text/event-stream"},
                )
                started = time.monotonic()
                limited = ""
                try:
                    resp = urllib.request.urlopen(req, timeout=self.timeout_s)
                except urllib.error.HTTPError as exc:
                    # 3a. Refused before a stream opened: by status and error code.
                    try:
                        error = json.loads(exc.read()).get("error") or {}
                    except ValueError:
                        error = {}
                    code = error.get("code") if isinstance(error, dict) else None
                    said = f"HTTP {exc.code} {code or ''}".strip()
                    if exc.code == 429 or code == "subscription_sharing_usage_limit_exceeded":
                        limited = said
                    elif exc.code in (401, 403) or code in _REFUSED:
                        with self._lock:
                            self._stopped = (
                                f"Refusing to run: the call was refused ({said}); {_SIGN_IN_AGAIN}."
                                if exc.code == 401 or code == "subscription_sharing_invalid_user"
                                else f"Refusing to run: the call was refused ({said}); no retry helps."
                            )
                        raise ModelUnavailable(self._stopped) from None
                    else:
                        raise RuntimeError(f"responses call failed: {said}") from None
                except OSError as exc:
                    raise RuntimeError(f"responses call failed: {exc}") from None

                if not limited:
                    # 3b. The stream, registered so close() can cut it.
                    with self._lock:
                        if self._pool is not session:
                            resp.close()
                            raise CancelledError()
                        self._running.add(resp)
                    text: list[str] = []
                    final: dict = {}
                    try:
                        for raw in resp:
                            line = raw.decode("utf-8").strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            event = json.loads(data)
                            kind = event.get("type")
                            if kind == "response.output_text.delta":
                                text.append(event.get("delta", ""))
                            elif kind == "response.output_item.added":
                                item = event.get("item") or {}
                                if item.get("type") not in _ANSWER_ITEMS:
                                    # The guard: whatever it is, the answer is not the prompt's alone.
                                    raise RuntimeError(
                                        f"the model used a tool ({item.get('type')}); answer discarded"
                                    )
                            elif kind in ("response.completed", "response.failed", "response.incomplete", "error"):
                                final = event
                                break
                            if time.monotonic() - started > self.timeout_s:
                                raise RuntimeError(f"responses call timed out after {self.timeout_s}s")
                    except (OSError, ValueError, AttributeError) as exc:
                        with self._lock:
                            if self._pool is not session:
                                raise CancelledError() from None
                        raise RuntimeError(f"responses stream broke: {exc}") from None
                    finally:
                        with self._lock:
                            self._running.discard(resp)
                        resp.close()
                    latency_ms = int((time.monotonic() - started) * 1000)

                    # 4. How the stream ended.
                    kind = final.get("type")
                    response = final.get("response") or {}
                    if kind == "response.completed":
                        break
                    error = response.get("error") or final.get("error") or {}
                    code = error.get("code") if isinstance(error, dict) else None
                    if code == "subscription_sharing_usage_limit_exceeded":
                        limited = code
                    elif code in _REFUSED:
                        with self._lock:
                            self._stopped = f"Refusing to run: the call was refused ({code}); no retry helps."
                        raise ModelUnavailable(self._stopped)
                    elif kind == "response.incomplete":
                        reason = (response.get("incomplete_details") or {}).get("reason", "")
                        raise RuntimeError(f"responses call incomplete: {reason}")
                    elif kind:
                        raise RuntimeError(f"responses call failed: {code or error}")
                    else:
                        raise RuntimeError("responses stream ended without a final event")

                # 5. A rate limit or the plan's usage limit: waited out, then a stop.
                if waits < self.rate_limit_waits:
                    wait_s = self.rate_limit_wait_s * 2**waits
                    log.warning("codex rate limited (%s); waiting %ss, then re-running", limited, wait_s)
                    time.sleep(wait_s)
                    waits += 1
                    continue
                # Still limited after every wait: a quota, which resets later.
                # Retrying it per prompt would only fail the rest of the run;
                # stopping leaves it resumable.
                with self._lock:
                    self._stopped = (
                        f"Refusing to run: still rate limited after {waits} waits ({limited}). "
                        "The quota resets later; close() the model and resume the run then."
                    )
                raise ModelUnavailable(self._stopped)

            # 6. The answer and what it cost.
            usage = response.get("usage") or {}
            generation = Generation(
                text="".join(text),
                latency_ms=latency_ms,
                usage=Usage(
                    prompt_tokens=int(usage.get("input_tokens") or 0),  # cached included
                    completion_tokens=int(usage.get("output_tokens") or 0),
                ),
                stop_reason=str(response.get("status") or "completed"),
                model=f"{response.get('model') or self.model}@responses",
            )
        except (ModelUnavailable, CancelledError):
            raise
        except Exception as exc:
            # 7. Count it. A run of failures with no success between them is
            # most likely one cause for all, so the model stops. A call cut by
            # close() is not counted, nor is one from before the last close().
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
