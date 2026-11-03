# Claude and Codex models and controllers

- **Status:** draft, agreed in a grilling session 2026-10-22, revised in a second one
  2026-10-26 (stopping on quota, process groups, resolved binary); built 2026-10-26
  (ADR 0002 v6). Codex moved off the CLI to the Responses API on the ChatGPT plan
  2026-10-28: see [Codex on the plan token](#codex-on-the-plan-token-agreed-and-built-2026-10-28-adr-0002-v8)
  (ADR 0002 v8)
- **Date:** 2026-10-22
- **Scope:** new `engine/models/claude_model.py`, `engine/models/codex_model.py`,
  `engine/controllers/claude_controller.py`, `engine/controllers/codex_controller.py`,
  `configs/claude.toml`, `configs/codex.toml`; one new field on `Generation` in
  `engine/models/base_model.py`; `CLAUDE_CONFIG_PATH` and `CODEX_CONFIG_PATH` in
  `engine/config.py`; new `tests/test_claude_model.py`,
  `tests/test_claude_controller.py`, `tests/test_codex_model.py`,
  `tests/test_codex_controller.py`; new `notebooks/15_claude_codex_models.ipynb`.

Governed by ADR 0002 (latest version). Mirrors `docs/specs/0001-qwen-model-controller.md`:
everything 0001 says about the controller holds here unless [Departures](#departures) says
otherwise. No service changes: which service drives these first (the 0011 pilot, or
`PurposeRiskService`) is that service's spec.

## The rule

**Same controller, different model.** `ClaudeController` and `CodexController` are copies
of `QwenController`. The only thing that changes is what runs a `Prompt`: a headless CLI
session on the user's subscription (Claude Code's `claude -p`, OpenAI's `codex exec`)
instead of `llama-server`. No Anthropic or OpenAI API key is used.

## Who owns what

| | Model (`ClaudeModel`, `CodexModel`) | Controller (`ClaudeController`, `CodexController`) |
|---|---|---|
| **Owns** | Execution: the subprocess, its flags, its temp folder, its pool; readiness; rate-limit waits | The model's existence; the in-flight budget, validation, parsing, retry, progress |
| **Knows** | The binary, the pinned model id and effort, how its CLI takes a system prompt, how its CLI reports errors and usage | `in_flight`, `max_attempts`, the checks by name, that a parser exists |
| **Never knows** | What a prompt is for; who is calling | That a CLI exists; what a label is; the answer format |

Each controller imports its model concretely: one controller per model (ADR 0002 v5). The
loop is copied from `QwenController`, not shared; a shared controller is for a later ADR
version if the three copies stay identical.

**The service provides the prompt.** As with Qwen, the service loads the prompt folder
(`repositories/prompts/<id>/`), builds `Prompt(system=…, user=…)` and hands it to the
controller. How `system` reaches the CLI is the model's business (`Prompt`'s docstring:
the wire format is the model's).

## The models

Both models implement `BaseModel`.

| verb | what it does |
|---|---|
| `ready()` | Idempotent. Resolves `binary` once (`expanduser()`, then `resolve()`: Claude's symlink becomes `~/.local/share/claude/versions/<v>`) and every later call runs that resolved file. The binary runs (`--version`, read here) and the CLI says it is logged in (`claude auth status` → `loggedIn: true`; `codex login status` → "Logged in"). No model call. Otherwise `ModelUnavailable`, naming the fix (`claude login`, `codex login`, or the toml line holding a binary path that no longer exists). Starts the pool. |
| `generate(prompt)` | One blocking subprocess in a fresh temp folder; returns text, token counts, latency, stop reason, model. See [The calls](#the-calls). |
| `fits(prompt)` | Always `True` (base default). Both models' contexts are far beyond any label. |
| `cache_shared_prefix(sample)` | Does nothing (base default). The providers cache on their side. |
| `submit(prompt)` | Queues one prompt into a pool of `n_parallel` workers, returns a `Future[Generation]` at once. |
| `close()` | Cancels queued prompts and kills running calls (their process groups); their answers are discarded. Clears a [tripped](#failures) state and the failure count. Idempotent. |

**Not one per process.** Unlike `QwenModel`, there is no singleton: a CLI model holds
nothing exclusive. Concurrency is bounded by each model's own pool.

**Isolated per call.** Each `generate()` runs in its own `tempfile.TemporaryDirectory()`
outside the repo, deleted afterwards, with no session saved. The CLI cannot load the repo's
CLAUDE.md or AGENTS.md, auto-memory, skills or past sessions, and has nothing on disk next
to it to read.

**One process group per call.** Both CLIs start child processes (Claude Code is Node;
Codex runs its own helpers). Each call starts with `start_new_session=True`, and a timeout
or `close()` kills the whole group with `killpg`, so no orphan keeps running and spending
quota. `close()` kills rather than waits: with `timeout_s = 300`, waiting could hold a
notebook interrupt for five minutes.

**Not deterministic.** Neither CLI exposes temperature, seed or an output-token cap. There
is no `[sampling]` table; `Prompt.max_tokens` and `Prompt.stop` are ignored, and each
model's docstring says so. `Prompt.schema` is not supported in this build (the purpose/risk
prompt uses the `lines` format, `schema=None`). A cut-off answer fails the prompt's `END`
check and is repaired like any failed check.

### The calls

**Claude** (`claude_model.py`). `Prompt.system` is written to `system.md` in the temp
folder; `Prompt.user` goes on stdin.

```
claude -p
  --model <[model].model>  --effort <[model].effort>
  --system-prompt-file <temp>/system.md
  --tools ""  --disable-slash-commands
  --strict-mcp-config  --setting-sources ""
  --no-session-persistence
  --output-format json
```

- `--system-prompt-file` **replaces** Claude Code's own system prompt: the model sees
  `Prompt.system` and `Prompt.user`, nothing else.
- `--strict-mcp-config --setting-sources ""` are required, not tidy-up: without them every
  call loaded the user's claude.ai connectors and settings, 202,253 input tokens for a
  ~1,600-token prompt.
- **Never `--bare`.** It reads auth only from `ANTHROPIC_API_KEY` / `apiKeyHelper`, never
  the subscription login.
- From the JSON object: `result` → `Generation.text`; `stop_reason`; `is_error` and
  `api_error_status` → [failures](#failures); `modelUsage[<pinned id>]` → usage and model.

**Codex** (`codex_model.py`). `Prompt.system` is written to `instructions.md` in the temp
folder; `Prompt.user` goes on stdin.

```
codex exec
  -m <[model].model>  -c model_reasoning_effort=<[model].effort>
  -c model_instructions_file='<temp>/instructions.md'
  -C <temp>  -s read-only
  --ephemeral  --skip-git-repo-check  --ignore-user-config  --ignore-rules
  <_STRIP: what Codex adds to the prompt, switched off>
  --json  -o <temp>/last.txt
  -
```

- `model_instructions_file` **replaces** Codex's base instructions. `base_instructions` and
  `experimental_instructions_file` are silently ignored on 0.162 (a warning appears only as
  an `error` item in the `--json` events) and the model then answers as a coding agent.
- What Codex adds beyond the prompt is switched off by `_STRIP` (2026-10-27), each switch
  checked with `codex debug prompt-input` on 0.164, which renders the model-visible input
  without a call. Off: the environment block (`include_environment_context`), the
  permissions text (`include_permissions_instructions`), the collaboration-mode text
  (`include_collaboration_mode_instructions`), the skills list
  (`skills.include_instructions`), the multi-agent role and its tools (`agents.enabled`),
  `request_user_input` (`tools.experimental_request_user_input.enabled`), web search, and
  the tools features bring (`--disable` shell_tool, unified_exec, apps, plugins,
  remote_plugin, sleep_tool, tool_suggest, view_image, image_generation). Not `code_mode_host`:
  with it off, `codex exec` refuses to start ("Code Mode is unavailable because code-mode host
  is disabled"), and it adds nothing to the prompt.
  Rendered context: ~37k chars before, ~10k after. What is left is the `exec` tool (a
  JavaScript cell holding only MCP-resource helpers and a clock) and its `wait`. No switch
  found removes them, and this is the one known difference from Claude's request. A key
  Codex does not know is silently ignored, so the switches are checked again when the
  binary moves.
- `--ignore-user-config` drops `~/.codex/config.toml`'s model and effort, so both are
  always passed explicitly.
- `last.txt` → `Generation.text`; `turn.completed.usage` → usage; `turn.failed` /
  `error` events → [failures](#failures).

**Tool guard (Codex).** No flag turns Codex's shell off, and `-s read-only` stops writes,
not reads. The model reads the `--json` events; if any command or file item appears, it
raises a request error (the answer is discarded), which the controller retries. Claude
needs no guard: `--tools ""` gives it no tools.

### What a generation records

| field | Claude | Codex |
|---|---|---|
| `text` | `result` | contents of `-o` file |
| `latency_ms` | wall time of the subprocess | wall time of the subprocess |
| `usage.prompt_tokens` | pinned model's `inputTokens + cacheCreationInputTokens + cacheReadInputTokens` | `input_tokens` (cached included) |
| `usage.completion_tokens` | pinned model's `outputTokens` | `output_tokens` |
| `stop_reason` | `stop_reason` | `""` (not reported) |
| `model` (new) | `<id from modelUsage>@claude-code-<version>` | `<pinned -m id>@codex-cli-<version>` |

`Usage` gets no new fields and raw JSON is not kept. Claude Code also makes a small Haiku
side call of its own (~1.2k tokens); it is not the pinned model's and is left out.

**New field `Generation.model: str = ""`** in `base_model.py`. Empty by default, so
`QwenModel` and its tests are unchanged. The CLI version comes from `<binary> --version`,
read once in `ready()`.

### Failures

| what happened | the model raises | what happens above |
|---|---|---|
| binary missing, or CLI not logged in | `ModelUnavailable` | the controller lets it through; the run stops |
| rate limit | nothing yet: the model waits `rate_limit_wait_s`, doubling, up to `rate_limit_waits` times, and re-runs the call itself | the controller sees one slow attempt |
| rate limit after the last wait | `ModelUnavailable` (the model trips) | the run stops; it is resumed after the quota resets |
| timeout (`timeout_s`), non-zero exit, `is_error` / `turn.failed`, a Codex tool event | the request's own error | the controller retries it within `max_attempts` |
| `max_consecutive_failures` calls in a row failed, of any kind, across all workers | `ModelUnavailable` (the model trips) | the run stops |

**A rate limit that outlasts the waits is a quota, not a hiccup.** A subscription cap
resets in hours; retrying it per label would spend ~21 minutes on each and write the rest
of the set as `error` outcomes. Stopping leaves a resumable run instead.

**The consecutive-failure limit** catches the same case while the rate-limit signals are
still unknown ([Open](#open) 1): a cap that is not recognised looks like a run of failed
requests. A call counts as failed only once it ends in an error; one that waited out a rate
limit and then answered is a success, and any success resets the count. Labels failing one
at a time still go through the controller's retries.

**Tripped stays tripped.** Once the model has raised `ModelUnavailable` for either
reason, every later `submit` fails with it at once, starting no subprocess, until `close()`
clears the state; `ready()` after `close()` makes the model usable again.

Rate-limit waits and the failure limit sit in the model, outside the controller's budget,
so the controller stays an exact copy of `QwenController`: it already re-raises
`ModelUnavailable` from a future.

## The controllers

As 0001's [controller section](0001-qwen-model-controller.md#the-controller), word for
word, with `QwenModel` → `ClaudeModel` / `CodexModel` and `configs/qwen.toml` →
`configs/claude.toml` / `configs/codex.toml`:

- `ClaudeController(model=None, max_attempts=None, in_flight=None)` builds `ClaudeModel()`
  when no model is given; same for Codex. `close()` closes the model.
- `stream(work, checks, parse, total, n_already_done, on_progress)` and
  `handle(prompts, checks, on_progress)`, with the same `Outcome` and `Attempt`.
- Same loop (peek, refill, finish, wait, settle one), same retries (failed checks and
  failed requests share `max_attempts`; a repair is built from the original prompt), same
  stopping, progress and errors.
- Each controller defines its own `ERROR` and `TOO_LONG` constants, as `qwen_controller.py`
  does. `TOO_LONG` never occurs, since `fits` is always `True`.

Every `Attempt` is kept in the `Outcome`, so a caller can count first-attempt failures
apart from final ones without new code.

## Checks

Unchanged: the checks and the parser come with the prompt
(`repositories/prompts/<id>/checks.py`, `parse.py`). The purpose/risk prompt's checks and
parser run on Claude and Codex answers as they are.

## Settings

`configs/claude.toml` and `configs/codex.toml`, committed, read through `engine/config.py`
like `qwen.toml`. Same precedence: constructor argument, then file, then hardcoded fallback.

| table | key | Claude | Codex |
|---|---|---|---|
| `[model]` | `binary` | `~/.local/bin/claude` (the auto-updating symlink) | the full path in the active VSCode extension, today `~/.vscode-server/extensions/openai.chatgpt-26.1021.22027-linux-x64/bin/linux-x86_64/codex` (codex-cli 0.164; 26.1007 held 0.162) |
| `[model]` | `model` | `claude-sonnet-5-5` | `gpt-6.1-terra` |
| `[model]` | `effort` | `medium` | `medium` |
| `[model]` | `n_parallel` | 8 | 8 |
| `[model]` | `timeout_s` | 300 | 300 |
| `[model]` | `rate_limit_wait_s`, `rate_limit_waits` | 60, 3 | 60, 3 |
| `[model]` | `max_consecutive_failures` | 16 (2 × `n_parallel`) | 16 (2 × `n_parallel`) |
| `[controller]` | `max_attempts` | 3 (as `qwen.toml`) | 3 (as `qwen.toml`) |
| `[controller]` | `in_flight` | 8 | 8 |

**Paths are pinned; CLI versions are recorded, not enforced.** Claude Code updates itself
(2.1.303 on 2026-10-19, 2.1.308 on 2026-10-22), and enforcing a version would fail
`ready()` every few days for a change that does not touch the pinned model. The version
goes into every `Generation.model` instead, and it is true for the whole run: `ready()`
resolves the symlink once and calls run that versioned file, so an update mid-run does not
switch binaries under a stale label (old versions stay on disk: 2.1.302, .303 and .308 on
2026-10-26). Both `binary` values go through `expanduser()`. The Codex path moves when VSCode updates the
extension; `ready()` then fails naming the toml line. No auto-discovery: sorting extension
folders by name is not version order, and a silent CLI switch is what the pinned path
prevents.

## Responsibilities

Rows as in 0001; only those that differ are listed.

| # | Responsibility | Model | Controller |
|---|---|---|---|
| 2 | Prompt format | Claude: `--system-prompt-file` + stdin. Codex: `model_instructions_file` + stdin | — |
| 3 | Constrained output | Not supported in this build | — |
| 4 | Lifecycle | A subprocess per call, a temp folder per call; nothing to free | Creates the model; readies it only once there is work; `close()` closes it |
| 5 | Readiness checks | Binary runs, CLI logged in | — |
| 6 | Health | Rate limit → wait and re-run; past the last wait, or `max_consecutive_failures` failures in a row → `ModelUnavailable`, tripped until `close()`; else the request's error | Lets `ModelUnavailable` through |
| 7 | Hosting and tuning | Flags that strip the CLI's own context (MCP, settings, tools, sessions) | — |
| 8 | Concurrency | At most `n_parallel` subprocesses at once | Keeps `in_flight` outstanding |
| 9 | Measuring the work | `fits` always true; no prefix cache | Unchanged (both calls are no-ops) |
| 10 | Settings | `[model]` | `[controller]` |
| 11 | Timeouts | `timeout_s` per call; the call's process group is killed | — |
| 22 | One instance | None: no singleton | Builds the model with `ClaudeModel()` / `CodexModel()` |
| 23 | Blindness | Empty temp folder, no session; Codex tool guard | — |
| 24 | Provenance | `Generation.model` = model id @ CLI version | — |

## Testing

Tests call only the public verbs; no test makes a real CLI call. The models' tests fake the
subprocess underneath: scripted stdout JSON (Claude) and `--json` event lines (Codex), exit
codes, a timeout, an `is_error` / `turn.failed` result, a rate-limit error (waits counted,
sleep faked), a Codex command event (guard), a binary that does not run and a CLI that is
not logged in (`ModelUnavailable`). Also: a rate limit past the last wait and
`max_consecutive_failures` failures in a row each raise `ModelUnavailable`; a success
resets the count; a tripped model fails the next `submit` without starting a subprocess and
works again after `close()` + `ready()`; a timeout and `close()` kill the process group;
calls run the resolved binary, not the symlink. They also check the argument list: the
stripping flags are present and `--bare` never is. The controllers' tests use a scripted `BaseModel`, as
`test_qwen_controller.py` does.

The live check is `notebooks/15_claude_codex_models.ipynb`, set up like
`07_qwen_model.ipynb`, which the user runs: one call per model through each controller's
`handle()` on the one-line passage, showing the `Outcome`, `Generation.model` and usage.

## Measurements behind the code

One-line passage "supports healthy joints. do not take if pregnant. keep out of reach of
children." with `prompts/purpose_risk/prompt.md`, 2026-10-22. Expected and returned by
every working call: `P: supports healthy joints` / `R: do not take if pregnant` / `END`.

| decision | measured |
|---|---|
| headless calls on the subscription, no API key | both CLIs answered; `claude auth status` → `authMethod: claude.ai`; `codex login status` → ChatGPT |
| `--strict-mcp-config --setting-sources ""` | Claude: 202,253 input tokens without them, 1,624 with them; 1.6 s |
| `model_instructions_file` for Codex | input tokens: `base_instructions` (ignored) 14,156 and a generic "what would you like me to do" reply; `experimental_instructions_file` (ignored) same; `developer_instructions` (adds on top) 14,675; `model_instructions_file` and `instructions` (replace) 10,591 |
| the same on codex-cli 0.164 (2026-10-26) | `model_instructions_file` still replaces: 11,171 input tokens, the expected three lines |
| `gpt-6-luna` | works on the ChatGPT account; `gpt-6-terra` is rejected there (400: "not supported when using Codex with a ChatGPT account") |
| `gpt-6.1-terra` (2026-10-28) | works on the ChatGPT account; replaces `gpt-6-luna`, which failed the answer checks on 4 of 50 pilot judge labels (Sonnet: 0) |
| version from `modelUsage` for Claude | `modelUsage` names `claude-sonnet-5-5` (and the Haiku side call separately); Codex reports no model |

## Codex on the plan token **[agreed and built 2026-10-28, ADR 0002 v8]**

Replaces the Codex half of [The calls](#the-calls), the Codex rows of the tables above, and
the Codex readiness and failure rules. Claude is unchanged.

**Why.** `codex exec` cannot be stripped bare: its `exec` and `wait` tools (9,510 chars,
~2,900 tokens) go with every call, and nothing turns them off. OpenAI's Sign in with
ChatGPT for open-source and locally hosted apps
(<https://developers.openai.com/siwc/token-sharing-open-source>) lets an app call the
ordinary Responses API with an OAuth token that bills the user's ChatGPT plan, not an API
key. Measured 2026-10-28, pilot label 239596, `gpt-6.1-terra`, effort medium: 2,180 input
tokens (2,036 instructions + 142 label + 2), no tools, 2.4 s, the same verdict as through
`codex exec` at 5,009 input tokens.

**What stays.** The class names `CodexModel` / `CodexController`, `configs/codex.toml`, the
public verbs (`ready`, `generate`, `submit`, `close`), the pool of `n_parallel` workers,
rate-limit waits, `max_consecutive_failures`, tripped-until-`close()`, and the controller,
which does not change at all. Nothing above the model knows the transport changed.

**New public verb `sign_in()`.** Interactive, once per machine (the refresh token then
lasts 30 days and renews itself). The OpenAI OAuth flow with PKCE: first sign-in uses
`client_id=dynamic_agent_client`, `agent_name_hint="supplement-kg judge"` and an
`ext_agent_host_id` (`urn:uuid:…`, created once and kept); scopes `openid profile email
offline_access resource.invoke chatgpt.tokens.use.direct`; `resource=https://api.openai.com/v1`;
redirect `http://127.0.0.1:1455/auth/callback`. It prints the authorize URL, waits up to 10
minutes on a threaded loopback listener (an idle connection must not block the callback:
VSCode's port forwarder opens one), exchanges the code at
`https://auth.openai.com/api/accounts/oauth/token`, checks `iss`, `aud`, `exp`, `nonce` and
that the scope holds `chatgpt.tokens.use.direct`, and stores the credentials. Called from a
notebook 15 cell. The ID token's signature is not checked: it comes straight from the token
endpoint over TLS (OIDC Core 3.1.3.7).

**Credentials** live in `[model].credentials_dir`, default
`~/.config/supplement-kg/chatgpt-plan/`, outside the repo: `host.json` (the host id),
`<client_id>.json` (tokens, scope, expiry, `sub`), `current.json` (which client is in use).
Directory `0700`, files `0600`, written atomically. Never logged, never in a `Generation`.

| verb | Codex now |
|---|---|
| `ready()` | Credentials exist and hold `chatgpt.tokens.use.direct`; the access token is refreshed if it expires within 5 minutes. Then one test call (instructions "Reply with OK.", input "OK", ~20 tokens): a model listing is a catalog, not an entitlement check, and only a completed call proves the plan accepts this client. The controller readies the model once there is work, so this is one call per run. Missing credentials, a refused refresh, or a test call refused with 401/403 → `ModelUnavailable`: "the ChatGPT sign-in has expired or was revoked; run `CodexModel().sign_in()`". Any other failed test call → `ModelUnavailable` naming the error. Starts the pool. |
| `generate(prompt)` | One streamed `POST https://api.openai.com/v1/responses` (below). Refreshes first if the token expires within 5 minutes. |
| `close()` | Cancels queued prompts; running calls stop reading their stream at the next event and are discarded. Clears a tripped state. |

**The request.** `urllib.request`, as `qwen_model.py`:

```
POST https://api.openai.com/v1/responses
Authorization: Bearer <access token>
{"model": <[model].model>, "instructions": Prompt.system,
 "input": [{"role": "user", "content": [{"type": "input_text", "text": Prompt.user}]}],
 "reasoning": {"effort": <[model].effort>}, "store": false, "stream": true}
```

No `tools`, so the model sees `Prompt.system` and `Prompt.user` and nothing else. The
stream's `response.output_text.delta` events build the text; `response.completed` ends a
success. `temperature`, `max_output_tokens` and the like are not accepted on this route, so
`Prompt.max_tokens`, `Prompt.stop` and `Prompt.schema` stay ignored.

**Refresh.** `POST` the token endpoint, form `grant_type=refresh_token`, the issued
`client_id`, `refresh_token`, `resource`; the replacement refresh token is stored with the
new access token. Serialized with an `fcntl` lock on the credentials file, so two notebook
kernels never race a rotating token; after taking the lock the file is read again.

**Tool guard.** Kept, though no tool is sent: an output item other than `message` or
`reasoning` fails the call.

| what happened | the model raises |
|---|---|
| no credentials, scope missing, refresh refused (`invalid_grant`, `refresh_token_expired`, `refresh_token_reused`, …) | `ModelUnavailable`: sign in again |
| HTTP 429 or `subscription_sharing_usage_limit_exceeded` | waits `rate_limit_wait_s`, doubling, `rate_limit_waits` times; still limited → `ModelUnavailable` (tripped) |
| 503 / `subscription_sharing_usage_unavailable` / `subscription_sharing_user_unavailable` | the request's own error; the controller retries |
| 401, 403, `subscription_sharing_user_not_eligible`, `subscription_sharing_unsupported_capability` | `ModelUnavailable` (no retry helps) |
| `response.failed` / `response.incomplete`, a stream cut short, timeout | the request's own error |

**A generation records** `text`; `latency_ms` (the request's wall time); `usage` from
`response.completed` (`input_tokens`, `output_tokens`); `stop_reason` `"completed"` or the
`incomplete_details.reason`; `model` = `<response model>@responses`.

**Settings.** `[model].binary` goes; `[model].credentials_dir` comes. The rest unchanged.

**Testing.** `urllib.request.urlopen` is faked: a scripted event stream, HTTP errors by code,
an expiring token (refresh called, file rewritten `0600`), a refused refresh
(`ModelUnavailable`), the usage-limit code (waits counted, then tripped), a tool item
(guard), the request body (no `tools`, `store: false`, `stream: true`, the
instructions as given), and `ready()`'s test call: one request, refused with 401 →
`ModelUnavailable` asking to sign in again, and not repeated by a second `ready()`. `sign_in()` is not unit-tested; notebook 15 is its live check.

**ADR.** ADR 0002 v7 says `codex_model.py` runs one CLI call per prompt; v8 says it calls the
Responses API on the user's ChatGPT plan.

## Open

1. **Rate-limit signals are not yet seen.** No call hit a limit. The models need to tell a
   rate limit from other failures (Claude: `api_error_status`; Codex: the error event's
   status); the exact values are to be confirmed on the first real one. Until then
   `max_consecutive_failures` is what stops a run that hits a cap.
2. **`in_flight` equals `n_parallel`.** 0001's rule is that `in_flight` exceeds
   `n_parallel` (~2x) so a freed worker always finds a queued prompt. With 4 and 4, a freed
   worker waits for the controller to refill. Agreed at 4 for now; 8 is the 0001-consistent
   value once rate limits are known.
3. **`timeout_s = 300`** is a starting value, not measured on long labels.

## Departures

From `QwenController`/`QwenModel` (spec 0001), on purpose:

1. **No singleton.** The models hold nothing exclusive.
2. **No sampling settings.** Answers are not deterministic.
3. **No `fits` limit, no prefix cache.** `TOO_LONG` never occurs.
4. **Rate-limit waits and a consecutive-failure limit in the model**, outside
   `max_attempts`; either one, exhausted, raises `ModelUnavailable` and the model stays
   tripped until `close()`.
5. **Codex tool guard.** A tool event makes the answer a failed request.
6. **`Generation.model`**, a new field on the shared shape, set by these two models only.
7. **No "Hello" check in `ready()` for Claude.** Readiness is binary + login; a model call
   there would spend subscription quota on every `ready()`. Codex on the plan token does make
   one (~20 tokens): its sign-in can expire or be revoked, and only a completed call shows it.
8. **`close()` kills, not waits.** Running calls are killed with their process group.
