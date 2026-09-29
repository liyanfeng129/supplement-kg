"""The entry to read a prompt by id.

A prompt is a folder here: `<id>/prompt.md` is the system text, `<id>/checks.py`
holds `CHECKS`, and `<id>/__init__.py` holds `STOP` and `MAX_TOKENS`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from engine.repositories.prompts.schema import PromptSpec

PROMPTS_DIR = Path(__file__).resolve().parent


def load(prompt_id: str) -> PromptSpec:
    """The system prompt `prompt_id` with its checks, stop sequences and budget."""
    folder = PROMPTS_DIR / prompt_id
    if not (folder / "prompt.md").is_file():
        known = sorted(p.parent.name for p in PROMPTS_DIR.glob("*/prompt.md"))
        raise KeyError(f"no prompt {prompt_id!r}; known: {known}")

    package = importlib.import_module(f"engine.repositories.prompts.{prompt_id}")
    checks = importlib.import_module(f"engine.repositories.prompts.{prompt_id}.checks")
    return PromptSpec(
        system=(folder / "prompt.md").read_text().rstrip("\n"),
        checks=tuple(checks.CHECKS),
        stop=tuple(package.STOP),
        max_tokens=package.MAX_TOKENS,
    )
