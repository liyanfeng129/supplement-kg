"""The purpose/risk answer format, read back into JSON fields.

The prompt asks for lines that are cheap to generate -- `P: claim` -- and
this turns them into what a row stores: `{"text": "claim"}` under `purpose`
for `P:` lines, under `risk` for `R:` lines, in answer order. The answer
carries no score: a separate scoring prompt judges a validated subset later.

Run by the controller on answers that passed every check in `checks.py`, so
each line here already has its prefix and a claim. Blank lines and a bare
`END` are skipped as the checks skip them; a line that still does not match
is skipped too rather than raised on -- the raw text stays in the row, so a
better parser can re-derive it.
"""

from __future__ import annotations

import re

_LINE = re.compile(r"([PR])\s*:(.*)")


def parse(text: str) -> dict:
    """`{"purpose": [...], "risk": [...]}`, each a list of `{"text"}`."""
    out: dict[str, list[dict]] = {"purpose": [], "risk": []}
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "END":
            continue
        match = _LINE.fullmatch(line)
        if not match:
            continue
        field = "purpose" if match.group(1) == "P" else "risk"
        out[field].append({"text": match.group(2).strip()})
    return out
