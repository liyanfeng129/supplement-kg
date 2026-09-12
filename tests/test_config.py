"""Config identity: what the hash covers, and what it must not.

Two runs pool only when their `config_hash` matches, so the hash has to cover
everything that makes results incomparable and nothing that does not. These
are cheap facts to assert and expensive ones to discover later, after a week of
runs have pooled that should not have.
"""

from __future__ import annotations

from dataclasses import replace

from engine.pipeline.config import Config, Decode


def test_the_prompt_has_exactly_one_source_of_truth():
    """`SYSTEM` must read the prompt file, not hold a second copy of it.

    The api path in `label_purpose_risk.extract()` and the local pipeline both
    send this text. A duplicated copy drifts silently: the pipeline would hash
    the file while `extract()` sent something else, and the gold set was built
    through that path.
    """
    from engine.services.extraction.label_purpose_risk import SYSTEM

    assert SYSTEM.strip() == Config().prompt.strip()


def test_editing_the_prompt_changes_the_hash(tmp_path):
    a = Config()
    edited = tmp_path / "prompt.md"
    edited.write_text(a.prompt + "\nOne more rule.\n")

    assert Config(prompt_path=str(edited)).config_hash != a.config_hash


def test_changing_a_decode_param_changes_the_hash():
    a = Config()
    b = Config(decode=replace(a.decode, temperature=0.7))

    assert a.config_hash != b.config_hash


def test_changing_the_model_changes_the_hash():
    a = Config()

    assert Config(model_id="some/other/model.gguf").config_hash != a.config_hash


def test_the_renderer_version_is_part_of_the_hash():
    """Editing what the model sees must invalidate the hash on its own."""
    assert "renderer_version" in Config().identity


def test_the_git_commit_is_not_part_of_the_hash():
    """Two genuinely identical runs a week apart must still pool."""
    assert "git_commit" not in Config().identity


def test_provenance_and_note_do_not_change_the_hash():
    """`provenance` is a column, not a branch -- and a note is free text."""
    a = Config()
    b = Config(provenance="api", note="tightened the boilerplate rule")

    assert a.config_hash == b.config_hash


def test_the_hash_is_invertible_into_the_config_it_names():
    """A hash must always be recoverable into the thing it names."""
    resolved = Config(note="baseline").resolved()

    assert resolved["config_hash"] == Config().config_hash
    # The prompt text travels verbatim: iteration is uncommitted and dead
    # prompt files get deleted, so a sha alone would not survive.
    assert resolved["prompt"] == Config().prompt
    assert resolved["note"] == "baseline"
    assert resolved["schema"]["required"]


def test_decode_records_enable_thinking_explicitly():
    """A silent flip of the model's default thinking mode must be visible."""
    assert Config().identity["decode"]["enable_thinking"] is False


def test_the_same_config_hashes_the_same_twice():
    assert Config().config_hash == Config().config_hash
