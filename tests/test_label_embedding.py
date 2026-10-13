"""LabelEmbeddingService: a label set's passages -> controller -> vectors beside the set.

What is pinned (spec 0008, acceptance 4 and 5): a run embeds every label of a
set, or the ids named, into an embedding named for its configuration, read
back in the set's or the given order; an empty passage is recorded `empty` and
never sent, and one too long is recorded `fits_context`; `ModelUnavailable`
stops the run with every finished vector on disk; a second run on the same name
embeds only what is not done, and refuses a changed instruction; the set's own
passages and manifest are unchanged.

No GPU: a real `EmbedController` drives a stand-in model that embeds a passage
by its words. The label sets live in `tmp_path`.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from engine.controllers.embed_controller import EmbedController
from engine.models.base_model import BaseEmbedModel, Embedding, ModelUnavailable, Usage
from engine.repositories.dsld.labels import loader as label_sets
from engine.repositories.dsld.labels import paths as label_paths
from engine.repositories.dsld.labels import store as label_store
from engine.repositories.prompts import loader as prompts
from engine.services.label_embedding.service import LabelEmbeddingService


def vector_of(text: str) -> np.ndarray:
    """A unit vector from the text alone, so any run gives a text the same row."""
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=8).astype(np.float32)
    return v / np.linalg.norm(v)


class ByText(BaseEmbedModel):
    """Embeds by text; records every text sent. A text starting `long` does not fit.

    `gone_after` makes requests after that many raise `ModelUnavailable`, as a
    server that died would.
    """

    def __init__(self, gone_after: int | None = None):
        self.gone_after = gone_after
        self.requests = 0
        self.sent: list[str] = []
        self.closed = False

    def fits(self, text, instruction=None):
        return not text.startswith("long")

    def embed(self, texts, instruction=None):
        self.requests += 1
        if self.gone_after is not None and self.requests > self.gone_after:
            raise ModelUnavailable("llama-server 2 exited with code -11")
        self.sent.extend(texts)
        return Embedding(
            vectors=np.stack([vector_of(t) for t in texts]),
            latency_ms=5,
            usage=Usage(prompt_tokens=10 * len(texts)),
        )

    def close(self):
        self.closed = True


PASSAGES = {
    11: "vitamin d3 softgels for bone health",
    12: "zinc lozenges for immune support",
    13: "",  # an empty passage
    14: "long passage that does not fit one pass",
    15: "magnesium glycinate for sleep",
    16: "fish oil omega-3 for heart health",
}


@pytest.fixture
def label_set(monkeypatch, tmp_path):
    """Label set `tiny` under tmp_path, with the passages above in id order."""
    monkeypatch.setattr(label_paths, "LABELS_ROOT", tmp_path / "labels")
    label_sets.reset()
    rows = [{"dsld_id": i, "name": f"label {i}", "passage": p} for i, p in PASSAGES.items()]
    label_store.write_label_set("tiny", rows, {"renderer_version": 4})
    yield "tiny"
    label_sets.reset()


def service(model: BaseEmbedModel, instruction: str | None = None) -> LabelEmbeddingService:
    instruction = prompts.load_instruction("label_clustering") if instruction is None else instruction
    return LabelEmbeddingService(EmbedController(model=model, instruction=instruction, batch=2, in_flight=2))


def failed_rows(label_set: str, name: str) -> dict[int, str]:
    p = label_paths.embedding_dir(label_set, name) / label_paths.FAILED_NAME
    return {r["dsld_id"]: r["failed"] for r in map(json.loads, p.read_text().splitlines())}


def test_a_run_embeds_every_label_of_the_set_in_its_order(label_set):
    model = ByText()

    with service(model) as svc:
        d = svc.run(label_set)

    name = d.name
    assert name.startswith("label_clustering-") and len(name) == len("label_clustering-") + 8
    ids, vectors = label_sets.load_embedding(label_set, name)
    assert ids.tolist() == [11, 12, 15, 16]
    assert np.allclose(vectors[0], vector_of(PASSAGES[11]))
    assert label_sets.read_embedding_manifest(label_set, name)["ids"] == list(PASSAGES)
    assert model.closed


def test_empty_and_too_long_passages_are_recorded_and_never_sent(label_set):
    model = ByText()

    with service(model) as svc:
        name = svc.run(label_set).name

    assert failed_rows(label_set, name) == {13: "empty", 14: "fits_context"}
    assert "" not in model.sent and PASSAGES[14] not in model.sent
    assert label_sets.embedded_ids(label_set, name) == set(PASSAGES)


def test_ids_pick_labels_in_the_order_given_under_an_explicit_name(label_set):
    with service(ByText()) as svc:
        svc.run(label_set, ids=[16, 11], name="two")

    assert label_sets.load_embedding(label_set, "two")[0].tolist() == [16, 11]


def test_ids_without_a_name_are_refused(label_set):
    with pytest.raises(ValueError, match="name"):
        service(ByText()).run(label_set, ids=[11])


def test_an_id_not_in_the_set_is_refused(label_set):
    with pytest.raises(ValueError, match="not in label set"):
        service(ByText()).run(label_set, ids=[11, 99], name="bad")


def test_a_dead_model_stops_the_run_with_every_finished_vector_on_disk(label_set):
    model = ByText(gone_after=1)

    with pytest.raises(ModelUnavailable):
        service(model).run(label_set, name="cut")

    ids, _ = label_sets.load_embedding(label_set, "cut")
    assert ids.tolist() == [11, 12]


def test_a_second_run_embeds_only_what_is_not_done(label_set):
    with pytest.raises(ModelUnavailable):
        service(ByText(gone_after=1)).run(label_set, name="cut")
    model = ByText()

    service(model).run(label_set, name="cut")

    assert model.sent == [PASSAGES[15], PASSAGES[16]]
    assert label_sets.load_embedding(label_set, "cut")[0].tolist() == [11, 12, 15, 16]


def test_a_finished_embedding_run_again_sends_nothing(label_set):
    name = service(ByText()).run(label_set).name
    model = ByText()

    assert service(model).run(label_set).name == name
    assert model.sent == []


def test_a_resume_under_a_changed_instruction_is_refused(label_set):
    service(ByText()).run(label_set, name="first")

    with pytest.raises(ValueError, match="config_hash"):
        service(ByText(), instruction="Find the brand").run(label_set, name="first")


def test_a_changed_instruction_gets_its_own_default_name(label_set):
    a = service(ByText()).run(label_set).name
    b = service(ByText(), instruction="Find the brand").run(label_set).name

    assert a != b


def test_the_label_set_s_passages_and_manifest_are_unchanged(label_set):
    d = label_paths.set_dir(label_set)
    before = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}

    service(ByText()).run(label_set)

    assert {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()} == before
