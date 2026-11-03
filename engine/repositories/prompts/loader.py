"""The entry to read a prompt by id.

A prompt is a folder here: `<id>/prompt.md` is the system text, `<id>/checks.py`
holds `CHECKS`, `<id>/parse.py` holds `parse`, and `<id>/__init__.py` holds
`STOP` and `MAX_TOKENS`.

An embedding instruction is a folder too, holding only `<id>/instruction.md`:
one sentence, and no Python files, because an embedding has no checks, parser,
stop sequences or budget.

A rulebook is `<id>/rulebook.md`: the rules a judge and the human both apply,
read as text (spec 0011.03).
"""

from __future__ import annotations

import importlib
from pathlib import Path

from engine.repositories.prompts.schema import PromptSpec

PROMPTS_DIR = Path(__file__).resolve().parent


def load(prompt_id: str) -> PromptSpec:
    """The system prompt `prompt_id` with its checks, parser, stop sequences and budget."""
    folder = PROMPTS_DIR / prompt_id
    if not (folder / "prompt.md").is_file():
        known = sorted(p.parent.name for p in PROMPTS_DIR.glob("*/prompt.md"))
        raise KeyError(f"no prompt {prompt_id!r}; known: {known}")

    package = importlib.import_module(f"engine.repositories.prompts.{prompt_id}")
    checks = importlib.import_module(f"engine.repositories.prompts.{prompt_id}.checks")
    parse = importlib.import_module(f"engine.repositories.prompts.{prompt_id}.parse")
    return PromptSpec(
        system=(folder / "prompt.md").read_text().rstrip("\n"),
        checks=tuple(checks.CHECKS),
        parse=parse.parse,
        stop=tuple(package.STOP),
        max_tokens=package.MAX_TOKENS,
    )


def load_instruction(prompt_id: str) -> str:
    """The embedding instruction `prompt_id`: the sentence in `<id>/instruction.md`."""
    path = PROMPTS_DIR / prompt_id / "instruction.md"
    if not path.is_file():
        known = sorted(p.parent.name for p in PROMPTS_DIR.glob("*/instruction.md"))
        raise KeyError(f"no instruction {prompt_id!r}; known: {known}")
    return path.read_text().rstrip("\n")


def load_rulebook(prompt_id: str) -> str:
    """The rulebook `prompt_id`: the text of `<id>/rulebook.md`."""
    path = PROMPTS_DIR / prompt_id / "rulebook.md"
    if not path.is_file():
        known = sorted(p.parent.name for p in PROMPTS_DIR.glob("*/rulebook.md"))
        raise KeyError(f"no rulebook {prompt_id!r}; known: {known}")
    return path.read_text().rstrip("\n")
