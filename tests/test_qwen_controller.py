"""QwenController runs every prompt to an outcome, batch by batch, retrying failures.

What is pinned: the controller submits what the model's plan names, in its
order, and re-submits only failures -- repaired to name the failing checks --
up to the budget; a prompt that never passes
comes back `ok=False` with the reasons and does not stall the run; a prompt the
plan calls too long is reported, never sent; outcomes line up with inputs by
position; progress counts units, not attempts; and a model that cannot be
readied stops the run before anything is sent.

The model here is a `QwenModel` subclass that answers from a script -- no GPU,
no server -- which keeps the controller's concrete dependency honest while
still testable.
"""

from __future__ import annotations

import pytest

from engine.controllers import qwen_controller as qc
from engine.controllers.qwen_controller import Outcome, QwenController
from engine.models.base_model import BaseModel, Generation, ModelUnavailable, Plan, Prompt, Usage
from engine.models.qwen_model import QwenModel


class ScriptedQwen(QwenModel):
    """Answers each prompt with the next scripted text; records what it saw.

    Plans in input order and answers each `submit()` before returning (the
    base model's default), so the script is consumed deterministically.
    """

    submit = BaseModel.submit

    def __init__(self, replies, fail_ready=None, too_long=()):
        super().__init__(model_path="/scripted.gguf")
        self.replies = list(replies)
        self.fail_ready = fail_ready
        self.too_long = tuple(too_long)
        self.seen: list[Prompt] = []
        self.readied = 0

    def ready(self) -> None:
        self.readied += 1
        if self.fail_ready:
            raise ModelUnavailable(self.fail_ready)

    def plan(self, prompts):
        order = tuple(i for i in range(len(prompts)) if i not in self.too_long)
        return Plan(order=order, too_long=self.too_long)

    def generate(self, prompt: Prompt) -> Generation:
        self.seen.append(prompt)
        return Generation(
            text=self.replies.pop(0),
            latency_ms=1000,
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            stop_reason="stop",
        )


def is_json(generation: Generation) -> bool:
    return generation.text.startswith("{")


def has_claims(generation: Generation) -> bool:
    return "claim" in generation.text


def prompt(user: str = "extract") -> Prompt:
    return Prompt(system="sys", user=user)


# --- the retry loop ---------------------------------------------------------


def test_a_passing_first_answer_is_one_attempt():
    model = ScriptedQwen(['{"claim": 1}'])

    [outcome] = QwenController(model).handle([prompt()], checks=[is_json])

    assert outcome == Outcome(
        generation=model_generation('{"claim": 1}'), ok=True, attempts=1, failed=[]
    )


def test_a_failed_check_is_named_in_the_re_sent_prompt():
    model = ScriptedQwen(["no json here", '{"claim": 1}'])

    [outcome] = QwenController(model).handle([prompt("extract")], checks=[is_json])

    assert outcome.ok and outcome.attempts == 2
    retry = model.seen[1]
    assert retry.user.startswith("extract")
    assert "is_json" in retry.user
    assert retry.system == "sys"


def test_the_repair_names_only_the_checks_that_failed():
    model = ScriptedQwen(['{"x": 1}', '{"claim": 1}'])

    QwenController(model).handle([prompt()], checks=[is_json, has_claims])

    assert "has_claims" in model.seen[1].user
    assert "is_json" not in model.seen[1].user


def test_each_repair_starts_from_the_original_prompt():
    """Repairs must not pile up: attempt 3 names attempt 2's failures only."""
    model = ScriptedQwen(["plain", '{"x": 1}', '{"claim": 1}'])

    QwenController(model).handle([prompt()], checks=[is_json, has_claims])

    third = model.seen[2].user
    assert "has_claims" in third
    assert "is_json" not in third


def test_it_gives_up_after_the_budget_and_says_why():
    model = ScriptedQwen(["a", "b", "c"])

    [outcome] = QwenController(model, max_attempts=3).handle(
        [prompt()], checks=[is_json]
    )

    assert not outcome.ok
    assert outcome.attempts == 3
    assert outcome.failed == ["is_json"]
    assert outcome.generation.text == "c"
    assert len(model.seen) == 3


def test_one_unsatisfiable_prompt_does_not_stall_the_batch():
    # Retries queue behind first attempts: one#1, two#1, one#2.
    model = ScriptedQwen(["bad", '{"claim": 2}', "bad"])

    outcomes = QwenController(model, max_attempts=2).handle(
        [prompt("one"), prompt("two")], checks=[is_json]
    )

    assert [o.ok for o in outcomes] == [False, True]
    assert outcomes[1].generation.text == '{"claim": 2}'


def test_without_checks_every_answer_passes():
    model = ScriptedQwen(["anything"])

    [outcome] = QwenController(model).handle([prompt()])

    assert outcome.ok and outcome.attempts == 1


def test_a_check_may_carry_its_own_name():
    class Named:
        name = "grounded"

        def __call__(self, _generation):
            return False

    model = ScriptedQwen(["x"])

    [outcome] = QwenController(model, max_attempts=1).handle(
        [prompt()], checks=[Named()]
    )

    assert outcome.failed == ["grounded"]


def test_outcomes_line_up_with_prompts_by_position():
    model = ScriptedQwen(["first", "second", "third"])

    outcomes = QwenController(model).handle([prompt(), prompt(), prompt()])

    assert [o.generation.text for o in outcomes] == ["first", "second", "third"]


# --- the model's plan -------------------------------------------------------


def test_prompts_are_submitted_in_the_models_plan_order():
    class Reversed(ScriptedQwen):
        def plan(self, prompts):
            return Plan(order=tuple(reversed(range(len(prompts)))))

    model = Reversed(["x", "y", "z"])

    QwenController(model).handle([prompt("0"), prompt("1"), prompt("2")])

    assert [p.user for p in model.seen] == ["2", "1", "0"]


def test_only_failures_are_re_submitted():
    model = ScriptedQwen(['{"a": 1}', "bad", '{"b": 2}'])

    outcomes = QwenController(model).handle([prompt("one"), prompt("two")], checks=[is_json])

    assert [o.attempts for o in outcomes] == [1, 2]
    assert [p.user.split("\n")[0] for p in model.seen] == ["one", "two", "two"]


def test_outcomes_follow_the_prompts_even_when_the_plan_reorders_them():
    class LongestFirst(ScriptedQwen):
        def plan(self, prompts):
            return Plan(order=(2, 0, 1))

    model = LongestFirst(["for 2", "for 0", "for 1"])

    outcomes = QwenController(model).handle([prompt(), prompt(), prompt()])

    assert [o.generation.text for o in outcomes] == ["for 0", "for 1", "for 2"]


def test_a_prompt_the_plan_calls_too_long_is_reported_not_sent():
    model = ScriptedQwen(["a", "c"], too_long=[1])

    outcomes = QwenController(model).handle([prompt("0"), prompt("1"), prompt("2")])

    assert [o.ok for o in outcomes] == [True, False, True]
    assert outcomes[1].failed == [qc.TOO_LONG]
    assert outcomes[1].attempts == 0
    assert len(model.seen) == 2


def test_a_plan_that_drops_a_prompt_is_an_error():
    class Forgetful(ScriptedQwen):
        def plan(self, prompts):
            return Plan(order=(0,))

    with pytest.raises(RuntimeError, match="unassigned"):
        QwenController(Forgetful(["a"])).handle([prompt(), prompt()])


# --- the budget -------------------------------------------------------------


def test_the_budget_comes_from_the_committed_file():
    assert QwenController(ScriptedQwen([])).max_attempts == 3


def test_a_budget_from_a_missing_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(qc, "DEFAULTS_PATH", tmp_path / "absent.toml")

    assert QwenController(ScriptedQwen([])).max_attempts == 3


def test_a_budget_that_is_not_an_integer_falls_back(monkeypatch, tmp_path):
    config = tmp_path / "qwen.toml"
    config.write_text('[controller]\nmax_attempts = "3"\n')
    monkeypatch.setattr(qc, "DEFAULTS_PATH", config)

    assert QwenController(ScriptedQwen([])).max_attempts == 3


def test_an_explicit_budget_wins():
    assert QwenController(ScriptedQwen([]), max_attempts=5).max_attempts == 5


def test_a_budget_below_one_is_refused():
    with pytest.raises(ValueError, match="max_attempts"):
        QwenController(ScriptedQwen([]), max_attempts=0)


# --- readiness --------------------------------------------------------------


def test_the_model_is_readied_before_anything_is_sent():
    model = ScriptedQwen(["x"])

    QwenController(model).handle([prompt()])

    assert model.readied >= 1


def test_an_unready_model_stops_the_run_with_a_readable_reason():
    model = ScriptedQwen(["x"], fail_ready="Refusing to run: only 6690 MiB free")

    with pytest.raises(ModelUnavailable) as caught:
        QwenController(model).handle([prompt(), prompt()])

    message = str(caught.value)
    assert "2 prompts" in message
    assert "6690 MiB" in message
    assert model.seen == []


def test_an_empty_batch_does_not_load_the_model():
    """A 40s load for zero prompts is pure cost."""
    model = ScriptedQwen([])

    assert QwenController(model).handle([]) == []
    assert model.readied == 0


# --- progress ---------------------------------------------------------------


def test_progress_counts_units_not_attempts():
    model = ScriptedQwen(["bad", '{"a": 1}', '{"b": 2}'])
    seen = []

    QwenController(model).handle(
        [prompt(), prompt()], checks=[is_json], on_progress=seen.append
    )

    assert [(p.done, p.total) for p in seen] == [(1, 2), (2, 2)]


def test_progress_detail_reports_tokens_and_what_is_left(monkeypatch):
    ticks = iter(range(100))  # the meter starts at 0; each unit done is 1s later
    monkeypatch.setattr(qc, "_now", lambda: float(next(ticks)))
    model = ScriptedQwen(["a", "b", "c", "d"])
    seen = []

    QwenController(model).handle([prompt()] * 4, on_progress=seen.append)

    first, last = seen[0].detail, seen[-1].detail
    assert "30 tok" in first  # one generation: 10 prompt + 20 completion
    assert "eta 3s" in first  # one second per unit so far, 3 units left
    assert "120 tok" in last
    assert "eta" not in last


# --- helpers ----------------------------------------------------------------


def model_generation(text: str) -> Generation:
    return Generation(
        text=text,
        latency_ms=1000,
        usage=Usage(prompt_tokens=10, completion_tokens=20),
        stop_reason="stop",
    )
