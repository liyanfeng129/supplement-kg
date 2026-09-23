"""Evidence preservation at the raw DSLD -> model input boundary."""

from engine.services.extraction.label_purpose_risk import label_segments, label_text
from engine.services.sources.dsld import parse_label


def test_metadata_is_omitted_but_misclassified_evidence_survives():
    label = parse_label({
        "id": 1, "fullName": "Product name", "brandName": "Brand name",
        "targetGroups": ["Adults"], "claims": ["Structure/Function"],
        "statements": [
            {"type": "FDA Disclaimer Statement", "notes": "Consult your physician."},
            {"type": "FDA Statement of Identity", "notes": "Antioxidant Supplement"},
            {"type": "General: Product/Version Code", "notes": "Do not exceed recommended intake."},
            {"type": "Storage", "notes": "Keep out of reach of children."},
            {"type": "Formulation re: Organic", "notes": "Supports energy."},
        ],
        "ingredientRows": [{"name": "Ingredient name"}],
    })
    passage = label_text(label)
    for statement in label.statements:
        assert statement["notes"] in passage
        assert statement["type"] not in passage
    for metadata in ("Product name", "Brand name", "Adults", "Structure/Function", "Ingredient name"):
        assert metadata not in passage


def test_nested_ingredient_text_and_duplicate_paths_survive():
    label = parse_label({
        "id": 2,
        "statements": [
            {"notes": "Supports  endurance."},
            {"type": "Brand IP", "notes": "Supports endurance."},
            {"notes": None},
        ],
        "ingredientRows": [{"nestedRows": [{"nestedRows": [{
            "notes": "Supports endurance.", "description": "Avoid during pregnancy.",
        }]}]}],
    })
    assert label_text(label) == "Supports endurance.\nAvoid during pregnancy."
    segments = label_segments(label)
    assert segments[0]["paths"] == [
        "statements[0].notes", "statements[1].notes",
        "ingredientRows[0].nestedRows[0].nestedRows[0].notes",
    ]
    assert segments[1]["paths"] == [
        "ingredientRows[0].nestedRows[0].nestedRows[0].description",
    ]


def test_no_text_produces_an_empty_passage():
    assert label_text(parse_label({"id": 3, "fullName": "Name only"})) == ""
