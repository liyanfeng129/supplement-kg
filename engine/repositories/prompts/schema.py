"""What the prompt repository hands out.

Beside the loader that fills it, as `dsld/schema.py` sits beside its loader.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.models.base_model import Check


@dataclass(frozen=True)
class PromptSpec:
    """A system prompt and everything its output format decides.

    The service adds only the user turn. Stop sequences, the output budget and
    the checks all follow from the format the prompt asks for, so they travel
    with it rather than being restated by each caller.
    """

    system: str
    checks: tuple[Check, ...]
    stop: tuple[str, ...]
    max_tokens: int
