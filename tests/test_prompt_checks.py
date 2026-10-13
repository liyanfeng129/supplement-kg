"""The prompt loader, and the purpose/risk checks against real answer shapes.

What is pinned: `load()` returns the prompt with its own checks, stop
sequences and budget; `load_instruction()` returns an embedding instruction's
one sentence; each check fails on exactly the rule it names and passes
everything else, including the empty answer a label with no claims produces.
"""

from __future__ import annotations

import pytest

from engine.models.base_model import Generation, Prompt
from engine.repositories.prompts import loader
from engine.repositories.prompts.purpose_risk import MAX_TOKENS, STOP
from engine.repositories.prompts.purpose_risk.checks import CHECKS

PROMPT = Prompt(system="sys", user="label text")
BY_NAME = {c.name: c for c in CHECKS}

GOOD = """P: supports memory retention
R: do not exceed recommended intake
R: consult your physicial prior to using this product if you are pregnant"""


def failed(text: str, stop_reason: str = "word") -> list[str]:
    generation = Generation(text=text, latency_ms=0, stop_reason=stop_reason)
    return [c.name for c in CHECKS if not c.passes(PROMPT, generation)]


# --- loader -----------------------------------------------------------------


def test_load_returns_the_prompt_with_everything_its_format_decides():
    spec = loader.load("purpose_risk")

    assert spec.system.startswith("You extract structured attributes")
    assert "P: <claim>\n" in spec.system
    assert "| " not in spec.system  # no score left in format or examples
    assert spec.checks == CHECKS
    assert spec.stop == STOP
    assert spec.max_tokens == MAX_TOKENS


def test_load_an_unknown_prompt_names_the_known_ones():
    with pytest.raises(KeyError, match="purpose_risk"):
        loader.load("no_such_prompt")


def test_load_instruction_returns_the_sentence_without_its_newline():
    assert loader.load_instruction("label_clustering") == (
        "Identify the main ingredients of the dietary supplement described by this label, "
        "the health purposes it claims, and the specific safety risks it warns about, "
        "such as side effects, drug interactions, medical conditions and groups who "
        "should not take it"
    )


def test_load_instruction_an_unknown_id_names_the_known_instructions():
    with pytest.raises(KeyError, match="label_clustering"):
        loader.load_instruction("purpose_risk")


# --- checks -----------------------------------------------------------------


def test_a_well_formed_answer_passes_every_check():
    assert failed(GOOD) == []


@pytest.mark.parametrize("text", ["", "END", "\n\n"])
def test_an_answer_with_no_claims_passes(text):
    assert failed(text) == []


def test_spacing_around_the_colon_is_tolerated():
    assert failed("P:supports sleep\nR : avoid alcohol") == []


def test_a_preamble_breaks_only_the_prefix_rule():
    assert failed("Here are the claims:\n" + GOOD) == ["line_prefix"]


def test_a_wrong_field_letter_breaks_only_the_prefix_rule():
    assert failed("purpose: supports memory retention") == ["line_prefix"]


def test_an_empty_claim_breaks_only_the_claim_rule():
    assert failed("P:  ") == ["claim_not_empty"]


def test_running_out_of_tokens_breaks_only_the_truncation_rule():
    assert failed(GOOD, stop_reason="limit") == ["not_truncated"]


def test_every_check_has_a_message_for_the_model():
    assert all(c.message.strip() for c in CHECKS)
    assert len(BY_NAME) == len(CHECKS)  # names are unique
