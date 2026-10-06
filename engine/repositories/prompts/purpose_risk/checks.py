"""The rules a purpose/risk answer must follow, one rule per check.

Each check looks at every line of the answer except blank lines and a bare
`END` (the server strips `\\nEND`, but an answer that *starts* with END keeps
it). An empty answer passes every check: a label stating no purpose and no
risk is a valid and common result.

The line checks are split so a failure names only what is wrong: a line with
no `P:` fails `line_prefix` alone. `claim_not_empty` only judges lines that
have the prefix, so it never repeats that failure.
"""

from __future__ import annotations

import re

from engine.models.base_model import Check, Generation, Prompt

_PREFIX = re.compile(r"[PR]\s*:")
_WHOLE = re.compile(r"[PR]\s*:(.*)")


def _line_prefix(prompt: Prompt, generation: Generation) -> bool:
    for line in generation.text.splitlines():
        line = line.strip()
        if not line or line == "END":
            continue
        if not _PREFIX.match(line):
            return False
    return True


def _claim_not_empty(prompt: Prompt, generation: Generation) -> bool:
    for line in generation.text.splitlines():
        line = line.strip()
        if not line or line == "END":
            continue
        whole = _WHOLE.fullmatch(line)
        if whole and not whole.group(1).strip():
            return False
    return True


def _not_truncated(prompt: Prompt, generation: Generation) -> bool:
    # llama-server reports "limit" when max_tokens ran out before a stop sequence.
    return generation.stop_reason != "limit"


CHECKS = (
    Check(
        name="line_prefix",
        passes=_line_prefix,
        message="Start every line with 'P: ' for a purpose or 'R: ' for a risk.",
    ),
    Check(
        name="claim_not_empty",
        passes=_claim_not_empty,
        message="Put the claim copied from the label after 'P: ' or 'R: '.",
    ),
    Check(
        name="not_truncated",
        passes=_not_truncated,
        message="Output only the P and R lines, then END. Nothing else.",
    ),
)
