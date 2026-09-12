"""The rendered prompt must match the model's own chat template.

`enable_thinking: false` is in the config hash because this model ships with
thinking on (`reasoning_effort: xhigh`) and a silent flip makes every result
incomparable. This llama-cpp-python has no `chat_template_kwargs` passthrough,
so the extractor renders the ChatML prompt itself -- which means the hand-rolled
string has to be checked against the template it is standing in for, not
against a reading of it.

Skipped when the weights are absent: the template lives in the GGUF.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.pipeline.config import Config, Decode
from engine.pipeline.extract import LocalExtractor

pytestmark = pytest.mark.skipif(
    not Path(Config().model_id).exists(), reason="model weights not on this machine"
)


def _template():
    from llama_cpp import Llama

    llm = Llama(model_path=Config().model_id, vocab_only=True, verbose=False)
    return llm.metadata["tokenizer.chat_template"]


def _render(template, **kwargs):
    from jinja2 import Environment

    def raise_exception(msg):
        raise ValueError(msg)

    env = Environment()
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
        add_generation_prompt=True,
        **kwargs,
    )


class _FixedPrompt(Config):
    """A config whose prompt text is fixed, so the comparison is about format."""

    @property
    def prompt(self) -> str:
        return "SYS"


def test_rendered_prompt_matches_the_template_with_thinking_off():
    extractor = LocalExtractor(_FixedPrompt())

    assert extractor.prompt_for("USER") == _render(_template(), enable_thinking=False)


def test_thinking_off_closes_the_think_block_immediately():
    """An open <think> would let the model spend its budget reasoning."""
    rendered = LocalExtractor(_FixedPrompt()).prompt_for("USER")

    assert rendered.endswith("<think>\n\n</think>\n\n")


def test_enable_thinking_true_leaves_the_think_block_open():
    cfg = _FixedPrompt(decode=Decode(enable_thinking=True))

    assert LocalExtractor(cfg).prompt_for("USER").endswith("<think>\n")


def test_enable_thinking_is_part_of_the_config_hash():
    """A silent flip must not be able to pool with results it is not comparable to."""
    off = Config(decode=Decode(enable_thinking=False))
    on = Config(decode=Decode(enable_thinking=True))

    assert off.config_hash != on.config_hash
