"""What the prompt repository hands out.

Beside the loader that fills it, as `dsld/schema.py` sits beside its loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from engine.models.base_model import Check


@dataclass(frozen=True)
class PromptSpec:
    """A system prompt and everything its output format decides.

    The service adds only the user turn. Stop sequences, the output budget,
    the checks and the parser all follow from the format the prompt asks for,
    so they travel with it rather than being restated by each caller.

    `parse` turns a passing answer's text into the fields a row stores; the
    controller runs it as it runs the checks, without knowing the format.
    """

    system: str
    checks: tuple[Check, ...]
    parse: Callable[[str], dict]
    stop: tuple[str, ...]
    max_tokens: int
