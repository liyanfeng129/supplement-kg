"""The purpose/risk parser: the short line format into JSON fields.

What is pinned: `P:` lines go to `purpose` and `R:` lines to `risk`, each as
`{"text"}` in answer order; a pipe is part of the claim, not a score
separator; spacing is tolerated; blank lines and a bare `END` are skipped; an empty answer is two empty lists; and `load()` hands the parser
out with the prompt.
"""

from __future__ import annotations

from engine.repositories.prompts import loader
from engine.repositories.prompts.purpose_risk.parse import parse


def test_p_and_r_lines_become_purpose_and_risk_terms():
    text = "P: supports restful sleep\nR: may cause drowsiness\nP: calms"

    assert parse(text) == {
        "purpose": [{"text": "supports restful sleep"}, {"text": "calms"}],
        "risk": [{"text": "may cause drowsiness"}],
    }


def test_an_answer_with_no_claims_is_two_empty_lists():
    for text in ("", "END", "\n\n"):
        assert parse(text) == {"purpose": [], "risk": []}


def test_a_pipe_stays_in_the_claim():
    assert parse("R: take with food | not on an empty stomach")["risk"] == [
        {"text": "take with food | not on an empty stomach"}
    ]


def test_spacing_around_the_colon_is_tolerated():
    assert parse("P:supports sleep\nR : avoid alcohol \n") == {
        "purpose": [{"text": "supports sleep"}],
        "risk": [{"text": "avoid alcohol"}],
    }


def test_blank_lines_and_a_trailing_end_are_skipped():
    assert parse("\nP: supports sleep\n\nEND\n")["purpose"] == [{"text": "supports sleep"}]


def test_load_hands_out_the_parser_with_the_prompt():
    assert loader.load("purpose_risk").parse is parse
