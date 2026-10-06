"""The cleaning service against the implementation the audits measured.

The selection rules carry 7,600 traced claims behind them (gold-1000 and
silver-1000 field audits). This service re-homes those rules; it does not
re-derive them, so the test that matters is equivalence: for every audited
label, the same passage, segment for segment and path for path. A difference
here is a regression against recorded evidence, not a style disagreement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.repositories.dsld.raw import loader as raw_loader
from engine.repositories.dsld.raw import paths as raw_paths
from engine.services.dsld_cleaning import RENDERER_VERSION, clean, flatten
from engine.services.extraction.label_purpose_risk import label_segments, label_text
from engine.services.sources.dsld import parse_label

ROOT = Path(__file__).resolve().parents[1]
AUDITED = (
    ROOT / "datasets" / "pilot" / "gold-1000.jsonl",
    ROOT / "datasets" / "pilot" / "silver-1000.jsonl",
)


def audited_ids(path: Path) -> list[int]:
    return [json.loads(line)["dsld_id"]
            for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("corpus", AUDITED, ids=lambda p: p.stem)
def test_matches_the_audited_renderer_on_every_label(corpus):
    """Same passage and same paths as `label_segments`, across the whole set."""
    if not corpus.exists() or not raw_paths.DUMP_ROOT.exists():
        pytest.skip(f"{corpus.name} or the DSLD dump is not present")

    for dsld_id in audited_ids(corpus):
        raw = raw_loader.load_raw(dsld_id)

        expected = label_segments(parse_label(raw))
        actual = clean(raw).segments

        assert [s["text"] for s in expected] == [s.text for s in actual], dsld_id
        assert [s["paths"] for s in expected] == [s.paths for s in actual], dsld_id
        assert label_text(parse_label(raw)) == flatten(clean(raw))["passage"], dsld_id


# --- the rules, stated one at a time ---------------------------------------

def test_only_the_allowlisted_paths_reach_the_passage():
    """Metadata and headers are absent; statement text of every type survives."""
    raw = {
        "id": 1, "fullName": "Product name", "brandName": "Brand name",
        "upcSku": "0123456789", "offMarket": 0, "entryDate": "2020-01-01",
        "targetGroups": ["Adults"], "claims": ["Structure/Function"],
        "statements": [
            {"type": "FDA Disclaimer Statement", "notes": "Consult your physician."},
            {"type": "FDA Statement of Identity", "notes": "Antioxidant Supplement"},
            {"type": "General: Product/Version Code", "notes": "Do not exceed intake."},
            {"type": "Seals/Symbols", "notes": "Supports weight loss."},
            {"type": "Storage", "notes": "Keep out of reach of children."},
        ],
        "ingredientRows": [{"name": "Ingredient name", "quantity": [{"quantity": 3, "unit": "mg"}]}],
    }
    passage = flatten(clean(raw))["passage"]

    for statement in raw["statements"]:
        assert statement["notes"] in passage
        assert statement["type"] not in passage
    for metadata in ("Brand name", "0123456789", "2020-01-01",
                     "Adults", "Structure/Function", "Ingredient name", "mg"):
        assert metadata not in passage


def test_the_product_name_is_both_carried_and_quotable():
    """The name is label text and often the only place a purpose is stated,
    so it is quotable -- under its own path, so it stays separable."""
    row = flatten(clean({"id": 2, "fullName": "Super Sleep Formula",
                         "statements": [{"type": "Other", "notes": "Supports sleep."}]}))
    assert row["name"] == "Super Sleep Formula"
    assert "Super Sleep Formula" in row["passage"]
    assert row["segments"][0]["paths"] == ["fullName"]


def test_a_name_only_label_still_yields_its_purpose():
    """DSLD 230595 ("Sleep Aid") in miniature: dosing and boilerplate in the
    statements, the purpose only in the name. Dropping the name loses it."""
    row = flatten(clean({"id": 8, "fullName": "Sleep Aid",
                         "statements": [{"type": "Other", "notes": "Take 2 capsules daily."}]}))
    assert "Sleep Aid" in row["passage"]


def test_nested_ingredient_text_survives_to_any_depth():
    raw = {
        "id": 3,
        "ingredientRows": [{
            "name": "Blend", "notes": "Top note.",
            "nestedRows": [{
                "name": "Inner", "description": "Inner description.",
                "nestedRows": [{"name": "Deepest", "notes": "Deepest note."}],
            }],
        }],
    }
    segments = clean(raw).segments
    assert [s.text for s in segments] == ["Top note.", "Inner description.", "Deepest note."]
    assert segments[-1].paths == ["ingredientRows[0].nestedRows[0].nestedRows[0].notes"]


def test_duplicate_text_merges_while_every_location_survives():
    raw = {"id": 4, "statements": [
        {"type": "Precautions", "notes": "Keep out of reach of children."},
        {"type": "Seals/Symbols", "notes": "Keep out of reach of children."},
    ]}
    segments = clean(raw).segments
    assert len(segments) == 1
    assert segments[0].paths == ["statements[0].notes", "statements[1].notes"]


def test_whitespace_collapses_so_grounding_tests_meaning_not_layout():
    raw = {"id": 5, "statements": [{"type": "Other", "notes": "Supports\n\n  restful   sleep."}]}
    assert clean(raw).segments[0].text == "Supports restful sleep."


def test_statements_precede_ingredients():
    raw = {
        "id": 6,
        "statements": [{"type": "Other", "notes": "Statement text."}],
        "ingredientRows": [{"name": "X", "notes": "Ingredient text."}],
    }
    assert [s.text for s in clean(raw).segments] == ["Statement text.", "Ingredient text."]


def test_a_label_with_no_free_text_cleans_to_an_empty_passage():
    """Empty, not an error: a label stating nothing is a real and common case.

    `fullName` is absent here, not merely blank -- with a name present the
    passage would hold it, which is the point of carrying the name at all.
    """
    row = flatten(clean({"id": 7, "statements": [],
                         "ingredientRows": [{"name": "X"}]}))
    assert row["passage"] == ""
    assert row["segments"] == []


def test_blank_and_missing_fields_are_skipped():
    raw = {"id": 8, "statements": [
        {"type": "A", "notes": None},
        {"type": "B", "notes": "   "},
        {"type": "C"},
        {"type": "D", "notes": "Real text."},
    ]}
    segments = clean(raw).segments
    assert [s.text for s in segments] == ["Real text."]
    assert segments[0].paths == ["statements[3].notes"]


# --- the seam ---------------------------------------------------------------

def test_one_or_many_follows_the_argument():
    one = {"id": 9, "statements": [{"type": "Other", "notes": "A."}]}
    assert clean(one).dsld_id == 9
    assert [c.dsld_id for c in clean([one, {**one, "id": 10}])] == [9, 10]
    assert isinstance(flatten(clean(one)), dict)
    assert isinstance(flatten(clean([one])), list)


def test_every_row_is_stamped_with_the_renderer_version():
    """A v2 passage file is otherwise indistinguishable from a v1 one on disk,
    and the gold set was labelled against v1."""
    assert flatten(clean({"id": 11}))["renderer_version"] == RENDERER_VERSION


def test_saved_rows_come_back_as_a_label_set(tmp_path, monkeypatch):
    from engine.repositories.dsld.labels import loader, paths
    from engine.services.dsld_cleaning import DsldCleaningService

    monkeypatch.setattr(paths, "LABELS_ROOT", tmp_path)
    loader.reset()
    rows = flatten(clean([{"id": 12, "fullName": "P",
                           "statements": [{"type": "Other", "notes": "Supports sleep."}]}]))

    sha = DsldCleaningService().save(rows, "one-r3")

    got_sha, labels = loader.load_flattened("one-r3")
    assert got_sha == sha
    assert labels[12].passage == rows[0]["passage"]
    manifest = json.loads((tmp_path / "one-r3" / paths.MANIFEST_NAME).read_text())
    assert (manifest["name"], manifest["renderer_version"]) == ("one-r3", RENDERER_VERSION)
    loader.reset()
