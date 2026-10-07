"""The cleaning service against the implementation the audits measured.

The selection rules carry 7,600 traced claims behind them (gold-1000 and
silver-1000 field audits). This service re-homes those rules; it does not
re-derive them. Renderer v4 (spec 0005) changes the *text* of each segment --
normalised, boilerplate cut -- but not which paths are read, so the test that
matters is that every audited label reads nothing the audited renderer did
not. A new path here is a regression against recorded evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.repositories.dsld.raw import loader as raw_loader
from engine.repositories.dsld.raw import paths as raw_paths
from engine.services.dsld_cleaning import RENDERER_VERSION, clean, flatten
from engine.services.extraction.label_purpose_risk import label_segments
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
def test_reads_only_the_audited_paths_on_every_label(corpus):
    """Every v4 path is one `label_segments` read, across the whole set.

    v4 drops boilerplate-only segments and merges segments that normalise
    alike, so it can have fewer paths -- never a new one. The one exception is
    `fullName`, added in v3 after the audits (see the module docstring of the
    service).
    """
    if not corpus.exists() or not raw_paths.DUMP_ROOT.exists():
        pytest.skip(f"{corpus.name} or the DSLD dump is not present")

    for dsld_id in audited_ids(corpus):
        raw = raw_loader.load_raw(dsld_id)

        audited = {p for s in label_segments(parse_label(raw)) for p in s["paths"]} | {"fullName"}
        actual = {p for s in clean(raw).segments for p in s.paths}

        assert actual <= audited, dsld_id


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
            {"type": "Storage", "notes": "Keep cool."},
        ],
        "ingredientRows": [{"name": "Ingredient name", "quantity": [{"quantity": 3, "unit": "mg"}]}],
    }
    passage = flatten(clean(raw))["passage"]

    for notes in ("consult your physician", "antioxidant supplement",
                  "do not exceed intake", "supports weight loss", "keep cool"):
        assert notes in passage
    for statement_type in ("disclaimer", "identity", "version code", "seals", "storage"):
        assert statement_type not in passage
    for metadata in ("brand name", "0123456789", "2020", "adults",
                     "structure function", "ingredient name", "mg"):
        assert metadata not in passage


def test_the_product_name_is_both_carried_and_quotable():
    """The name is label text and often the only place a purpose is stated,
    so it is quotable -- under its own path, so it stays separable."""
    row = flatten(clean({"id": 2, "fullName": "Super Sleep Formula",
                         "statements": [{"type": "Other", "notes": "Supports sleep."}]}))
    assert row["name"] == "Super Sleep Formula"
    assert "super sleep formula" in row["passage"]
    assert row["segments"][0]["paths"] == ["fullName"]


def test_a_name_only_label_still_yields_its_purpose():
    """DSLD 230595 ("Sleep Aid") in miniature: dosing and boilerplate in the
    statements, the purpose only in the name. Dropping the name loses it."""
    row = flatten(clean({"id": 8, "fullName": "Sleep Aid",
                         "statements": [{"type": "Other", "notes": "Take 2 capsules daily."}]}))
    assert "sleep aid" in row["passage"]


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
    assert [s.text for s in segments] == ["top note", "inner description", "deepest note"]
    assert segments[-1].paths == ["ingredientRows[0].nestedRows[0].nestedRows[0].notes"]


def test_duplicate_text_merges_while_every_location_survives():
    raw = {"id": 4, "statements": [
        {"type": "Precautions", "notes": "Avoid if allergic to shellfish."},
        {"type": "Seals/Symbols", "notes": "Avoid if allergic to shellfish."},
    ]}
    segments = clean(raw).segments
    assert len(segments) == 1
    assert segments[0].paths == ["statements[0].notes", "statements[1].notes"]


def test_whitespace_collapses_so_grounding_tests_meaning_not_layout():
    raw = {"id": 5, "statements": [{"type": "Other", "notes": "Supports\n\n  restful   sleep."}]}
    assert clean(raw).segments[0].text == "supports restful sleep"


def test_statements_precede_ingredients():
    raw = {
        "id": 6,
        "statements": [{"type": "Other", "notes": "Statement text."}],
        "ingredientRows": [{"name": "X", "notes": "Ingredient text."}],
    }
    assert [s.text for s in clean(raw).segments] == ["statement text", "ingredient text"]


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
    assert [s.text for s in segments] == ["real text"]
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
    assert RENDERER_VERSION == 4
    assert flatten(clean({"id": 11}))["renderer_version"] == RENDERER_VERSION


def test_saved_rows_come_back_as_a_label_set(tmp_path, monkeypatch):
    from engine.repositories.dsld.labels import loader, paths
    from engine.services.dsld_cleaning import DsldCleaningService

    monkeypatch.setattr(paths, "LABELS_ROOT", tmp_path)
    loader.reset()
    rows = flatten(clean([{"id": 12, "fullName": "P",
                           "statements": [{"type": "Other", "notes": "Supports sleep."}]}]))

    sha = DsldCleaningService().save(rows, "one-r4")

    got_sha, labels = loader.load_flattened("one-r4")
    assert got_sha == sha
    assert labels[12].passage == rows[0]["passage"]
    manifest = json.loads((tmp_path / "one-r4" / paths.MANIFEST_NAME).read_text())
    assert (manifest["name"], manifest["renderer_version"]) == ("one-r4", RENDERER_VERSION)
    loader.reset()


# --- renderer v4: normalising and the checklist (spec 0005) -----------------

def passage_of(*notes: str) -> str:
    raw = {"id": 20, "statements": [{"type": "Other", "notes": n} for n in notes]}
    return flatten(clean(raw))["passage"]


def test_case_punctuation_and_amounts_do_not_tell_copies_apart():
    assert passage_of("Vitamin C.") == passage_of("vitamin c")
    assert passage_of("500 mg Vitamin C per serving") == passage_of("1,000 mg vitamin C per serving")


def test_digits_that_are_part_of_a_name_survive_and_stay_distinct():
    passage = passage_of("B6", "B12", "Omega-3", "Omega-6", "CoQ10", "5-HTP")
    assert passage.split("\n") == ["b6", "b12", "omega 3", "omega 6", "coq10", "5 htp"]


def test_age_limits_survive():
    passage = passage_of("Not for use by persons under the age of 18.",
                         "Do not give to children under 12.", "For ages 4 and up.")
    assert "under the age of 18" in passage
    assert "children under 12" in passage
    assert "ages 4 and up" in passage


def test_a_pure_boilerplate_segment_is_dropped():
    raw = {"id": 21, "statements": [
        {"type": "Storage", "notes": "Keep out of reach of children."},
        {"type": "Other", "notes": "Supports sleep."},
    ]}
    segments = clean(raw).segments
    assert [s.text for s in segments] == ["supports sleep"]
    assert segments[0].paths == ["statements[1].notes"]


def test_a_mixed_segment_keeps_its_real_caution():
    assert (passage_of("Keep out of reach of children. Discontinue use if rapid heartbeat occurs.")
            == "discontinue use if rapid heartbeat occurs")


def test_the_longest_checklist_phrase_is_cut_whole():
    """Both phrases are on the checklist; the shorter must not leave `warning` behind."""
    assert passage_of("Warning: Keep out of reach of children.") == ""


def test_checklist_phrases_in_the_product_name_are_not_cut():
    row = flatten(clean({"id": 22, "fullName": "Dietary Supplement Sleep Formula"}))
    assert row["passage"] == "dietary supplement sleep formula"


def test_segments_that_normalise_alike_merge_with_both_paths():
    segments = clean({"id": 23, "statements": [
        {"type": "A", "notes": "Take 2 capsules with food."},
        {"type": "B", "notes": "take 4 capsules, with food"},
    ]}).segments
    assert [(s.text, s.paths) for s in segments] == [
        ("take with food", ["statements[0].notes", "statements[1].notes"])]


def test_an_all_boilerplate_label_keeps_one_empty_row():
    row = flatten(clean({"id": 24, "statements": [
        {"type": "A", "notes": "Keep out of reach of children."},
        {"type": "B", "notes": "Store in a cool, dry place."},
    ]}))
    assert (row["dsld_id"], row["passage"], row["segments"]) == (24, "", [])


def test_the_row_name_stays_raw_while_the_passage_is_normalised():
    row = flatten(clean({"id": 25, "fullName": "Vitamin D3 2,000 IU"}))
    assert row["name"] == "Vitamin D3 2,000 IU"
    assert row["passage"] == "vitamin d3"
    assert row["renderer_version"] == 4


def test_a_checklist_phrase_written_with_numbers_still_matches():
    """`store at 30 c 86 f` is on the checklist as mined; normalising it at load
    makes it match the line it came from."""
    assert passage_of("Store at 30°C (86°F).") == ""
