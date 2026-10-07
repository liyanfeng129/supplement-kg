#!/usr/bin/env bash
# Stop hook: when the uncommitted diff under engine/ is new, block once and ask
# the agent to run the architecture-reviewer subagent on it (ADR 0002).
#
# Lets the agent stop when: the hook already blocked this turn
# (stop_hook_active), nothing under engine/ changed, or this exact diff was
# already reviewed (hash in .claude/state/, gitignored).
set -euo pipefail

input=$(cat)
if printf '%s' "$input" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("stop_hook_active") else 1)'; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

untracked=$(git ls-files --others --exclude-standard -- engine/ | grep -v '__pycache__' || true)
changed=$(git diff HEAD --name-only -- engine/; printf '%s\n' "$untracked")
if [ -z "$(printf '%s' "$changed" | tr -d '[:space:]')" ]; then
  exit 0
fi

state=.claude/state/reviewed-engine-diff
hash=$( { git diff HEAD -- engine/; for f in $untracked; do printf '%s\n' "$f"; cat "$f"; done; } | sha256sum | cut -d' ' -f1)
if [ -f "$state" ] && [ "$(cat "$state")" = "$hash" ]; then
  exit 0
fi
mkdir -p .claude/state
printf '%s' "$hash" > "$state"

python=.venv/bin/python
[ -x "$python" ] || python=python3
layers=$("$python" -m pytest -q tests/test_layers.py 2>&1 | tail -n 30 || true)

cat >&2 <<MSG
engine/ has unreviewed changes:
$(printf '%s\n' "$changed" | sed '/^$/d; s/^/  /')

tests/test_layers.py:
$(printf '%s\n' "$layers" | sed 's/^/  /')

Before finishing, run the architecture-reviewer subagent (Agent tool,
subagent_type "architecture-reviewer") on the uncommitted diff under engine/.
Then:
- mechanical findings (upward import, folder name, file in the wrong layer by
  rule, missing ADR version): fix them now.
- design findings (moving or splitting a function, a function no spec agreed):
  do not restructure; report them to the user with the reviewer's reasoning.
MSG
exit 2
