"""The purpose/risk parser: the short line format into JSON fields.

What is pinned: `P:` lines go to `purpose` and `R:` lines to `risk`, each as
`{"text", "score"}` in answer order; the score binds to the last pipe, so a
claim may contain one; spacing is tolerated; blank lines and a bare `END` are
skipped; an empty answer is two empty lists; and `load()` hands the parser
out with the prompt.
"""

from __future__ import annotations

from engine.repositories.prompts import loader
from engine.repositories.prompts.purpose_risk.parse import parse


def test_p_and_r_lines_become_purpose_and_risk_terms():
    text = "P: supports restful sleep | 4\nR: may cause drowsiness | 3\nP: calms | 2"

    assert parse(text) == {
        "purpose": [
            {"text": "supports restful sleep", "score": 4},
            {"text": "calms", "score": 2},
        ],
        "risk": [{"text": "may cause drowsiness", "score": 3}],
    }


def test_an_answer_with_no_claims_is_two_empty_lists():
    for text in ("", "END", "\n\n"):
        assert parse(text) == {"purpose": [], "risk": []}


def test_the_score_binds_to_the_last_pipe():
    assert parse("R: take with food | not on an empty stomach | 3")["risk"] == [
        {"text": "take with food | not on an empty stomach", "score": 3}
    ]


def test_spacing_around_colon_and_pipe_is_tolerated():
    assert parse("P:supports sleep|4\nR : avoid alcohol |  2\n") == {
        "purpose": [{"text": "supports sleep", "score": 4}],
        "risk": [{"text": "avoid alcohol", "score": 2}],
    }


def test_blank_lines_and_a_trailing_end_are_skipped():
    assert parse("\nP: supports sleep | 4\n\nEND\n")["purpose"] == [
        {"text": "supports sleep", "score": 4}
    ]


def test_load_hands_out_the_parser_with_the_prompt():
    assert loader.load("purpose_risk").parse is parse
