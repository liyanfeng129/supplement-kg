# Claude and Codex models and controllers

- **Status:** draft, agreed in a grilling session 2026-10-22; not built
- **Date:** 2026-10-22
- **Scope:** new `engine/models/claude_model.py`, `engine/models/codex_model.py`,
  `engine/controllers/claude_controller.py`, `engine/controllers/codex_controller.py`,
  `configs/claude.toml`, `configs/codex.toml`; one new field on `Generation` in
  `engine/models/base_model.py`; new `tests/test_claude_model.py`,
  `tests/test_claude_controller.py`, `tests/test_codex_model.py`,
  `tests/test_codex_controller.py`.

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
| `ready()` | Idempotent. The binary runs (`--version`) and the CLI says it is logged in (`claude auth status` → `loggedIn: true`; `codex login status` → "Logged in"). No model call. Otherwise `ModelUnavailable`, naming the fix (`claude login`, `codex login`, or the toml line holding a binary path that no longer exists). Starts the pool. |
| `generate(prompt)` | One blocking subprocess in a fresh temp folder; returns text, token counts, latency, stop reason, model. See [The calls](#the-calls). |
| `fits(prompt)` | Always `True` (base default). Both models' contexts are far beyond any label. |
| `cache_shared_prefix(sample)` | Does nothing (base default). The providers cache on their side. |
| `submit(prompt)` | Queues one prompt into a pool of `n_parallel` workers, returns a `Future[Generation]` at once. |
| `close()` | Cancels queued prompts, waits for running subprocesses. Idempotent. |

**Not one per process.** Unlike `QwenModel`, there is no singleton: a CLI model holds
nothing exclusive. Concurrency is bounded by each model's own pool.

**Isolated per call.** Each `generate()` runs in its own `tempfile.TemporaryDirectory()`
outside the repo, deleted afterwards, with no session saved. The CLI cannot load the repo's
CLAUDE.md or AGENTS.md, auto-memory, skills or past sessions, and has nothing on disk next
to it to read.

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
  --json  -o <temp>/last.txt
  -
```

- `model_instructions_file` **replaces** Codex's base instructions. `base_instructions` and
  `experimental_instructions_file` are silently ignored on 0.162 (a warning appears only as
  an `error` item in the `--json` events) and the model then answers as a coding agent.
- What Codex still sends beyond the prompt: its tool definitions (~10k tokens). No flag
  removes them. This is the one known difference from Claude's request.
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
| rate limit after the last wait | the request's own error | the controller retries it within `max_attempts` |
| timeout (`timeout_s`), non-zero exit, `is_error` / `turn.failed`, a Codex tool event | the request's own error | the controller retries it within `max_attempts` |

Rate-limit waits sit in the model, outside the controller's budget, so the controller stays
an exact copy of `QwenController`.

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
| `[model]` | `binary` | `~/.local/bin/claude` (the auto-updating symlink) | the full path in the active VSCode extension, today `~/.vscode-server/extensions/openai.chatgpt-26.1007.21434-linux-x64/bin/linux-x86_64/codex` |
| `[model]` | `model` | `claude-sonnet-5-5` | `gpt-6-luna` |
| `[model]` | `effort` | `medium` | `medium` |
| `[model]` | `n_parallel` | 4 | 4 |
| `[model]` | `timeout_s` | 300 | 300 |
| `[model]` | `rate_limit_wait_s`, `rate_limit_waits` | 60, 3 | 60, 3 |
| `[controller]` | `max_attempts` | 3 (as `qwen.toml`) | 3 (as `qwen.toml`) |
| `[controller]` | `in_flight` | 4 | 4 |

**Paths are pinned; CLI versions are recorded, not enforced.** Claude Code updates itself
(2.1.303 on 2026-10-19, 2.1.308 on 2026-10-22), and enforcing a version would fail
`ready()` every few days for a change that does not touch the pinned model. The version
goes into every `Generation.model` instead. The Codex path moves when VSCode updates the
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
| 6 | Health | Rate limit → wait and re-run; else the request's error | — |
| 7 | Hosting and tuning | Flags that strip the CLI's own context (MCP, settings, tools, sessions) | — |
| 8 | Concurrency | At most `n_parallel` subprocesses at once | Keeps `in_flight` outstanding |
| 9 | Measuring the work | `fits` always true; no prefix cache | Unchanged (both calls are no-ops) |
| 10 | Settings | `[model]` | `[controller]` |
| 11 | Timeouts | `timeout_s` per call | — |
| 22 | One instance | None: no singleton | Builds the model with `ClaudeModel()` / `CodexModel()` |
| 23 | Blindness | Empty temp folder, no session; Codex tool guard | — |
| 24 | Provenance | `Generation.model` = model id @ CLI version | — |

## Testing

Tests call only the public verbs; no test makes a real CLI call. The models' tests fake the
subprocess underneath: scripted stdout JSON (Claude) and `--json` event lines (Codex), exit
codes, a timeout, an `is_error` / `turn.failed` result, a rate-limit error (waits counted,
sleep faked), a Codex command event (guard), a binary that does not run and a CLI that is
not logged in (`ModelUnavailable`). They also check the argument list: the stripping flags
are present and `--bare` never is. The controllers' tests use a scripted `BaseModel`, as
`test_qwen_controller.py` does.

The live check is a notebook cell the user runs: one call per model on a one-line passage.

## Measurements behind the code

One-line passage "supports healthy joints. do not take if pregnant. keep out of reach of
children." with `prompts/purpose_risk/prompt.md`, 2026-10-22. Expected and returned by
every working call: `P: supports healthy joints` / `R: do not take if pregnant` / `END`.

| decision | measured |
|---|---|
| headless calls on the subscription, no API key | both CLIs answered; `claude auth status` → `authMethod: claude.ai`; `codex login status` → ChatGPT |
| `--strict-mcp-config --setting-sources ""` | Claude: 202,253 input tokens without them, 1,624 with them; 1.6 s |
| `model_instructions_file` for Codex | input tokens: `base_instructions` (ignored) 14,156 and a generic "what would you like me to do" reply; `experimental_instructions_file` (ignored) same; `developer_instructions` (adds on top) 14,675; `model_instructions_file` and `instructions` (replace) 10,591 |
| `gpt-6-luna` | works on the ChatGPT account; `gpt-6-terra` is rejected there (400: "not supported when using Codex with a ChatGPT account") |
| version from `modelUsage` for Claude | `modelUsage` names `claude-sonnet-5-5` (and the Haiku side call separately); Codex reports no model |

## Open

1. **Rate-limit signals are not yet seen.** No call hit a limit. The models need to tell a
   rate limit from other failures (Claude: `api_error_status`; Codex: the error event's
   status); the exact values are to be confirmed on the first real one.
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
4. **Rate-limit waits in the model**, outside `max_attempts`.
5. **Codex tool guard.** A tool event makes the answer a failed request.
6. **`Generation.model`**, a new field on the shared shape, set by these two models only.
7. **No "Hello" check in `ready()`.** Readiness is binary + login; a model call there would
   spend subscription quota on every `ready()`.
