"""The prompt loader, and the purpose/risk checks against real answer shapes.

What is pinned: `load()` returns the prompt with its own checks, stop
sequences and budget; each check fails on exactly the rule it names and passes
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

GOOD = """P: supports memory retention | 5
R: do not exceed recommended intake | 5
R: consult your physicial prior to using this product if you are pregnant | 4"""


def failed(text: str, stop_reason: str = "word") -> list[str]:
    generation = Generation(text=text, latency_ms=0, stop_reason=stop_reason)
    return [c.name for c in CHECKS if not c.passes(PROMPT, generation)]


# --- loader -----------------------------------------------------------------


def test_load_returns_the_prompt_with_everything_its_format_decides():
    spec = loader.load("purpose_risk")

    assert spec.system.startswith("You extract structured attributes")
    assert "P: <claim> | <1-5>" in spec.system
    assert spec.checks == CHECKS
    assert spec.stop == STOP
    assert spec.max_tokens == MAX_TOKENS


def test_load_an_unknown_prompt_names_the_known_ones():
    with pytest.raises(KeyError, match="purpose_risk"):
        loader.load("no_such_prompt")


# --- checks -----------------------------------------------------------------


def test_a_well_formed_answer_passes_every_check():
    assert failed(GOOD) == []


@pytest.mark.parametrize("text", ["", "END", "\n\n"])
def test_an_answer_with_no_claims_passes(text):
    assert failed(text) == []


def test_a_claim_may_contain_a_pipe():
    assert failed("R: take with food | not on an empty stomach | 3") == []


def test_spacing_around_colon_and_pipe_is_tolerated():
    assert failed("P:supports sleep|4\nR : avoid alcohol |  2") == []


def test_a_preamble_breaks_both_ends_of_the_line():
    assert failed("Here are the claims:\n" + GOOD) == ["line_prefix", "line_score"]


def test_a_missing_score_breaks_only_the_score_rule():
    assert failed("P: supports memory retention") == ["line_score"]


@pytest.mark.parametrize("score", ["0", "6", "a"])
def test_a_score_outside_1_to_5_breaks_the_score_rule(score):
    assert failed(f"P: supports memory retention | {score}") == ["line_score"]


def test_a_wrong_field_letter_breaks_only_the_prefix_rule():
    assert failed("purpose: supports memory retention | 5") == ["line_prefix"]


def test_an_empty_claim_breaks_only_the_claim_rule():
    assert failed("P:  | 5") == ["claim_not_empty"]


def test_running_out_of_tokens_breaks_only_the_truncation_rule():
    assert failed(GOOD, stop_reason="limit") == ["not_truncated"]


def test_every_check_has_a_message_for_the_model():
    assert all(c.message.strip() for c in CHECKS)
    assert len(BY_NAME) == len(CHECKS)  # names are unique
