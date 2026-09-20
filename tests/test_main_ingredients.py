"""Selection edge cases: no main ingredient inferred from unavailable amounts."""

import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "datasets/pilot/find_main_ingredients.py"


def amount(value, unit="mg", basis=1, operator="="):
    return {"quantity": value, "unit": unit, "operator": operator,
            "servingSizeOrder": 1, "servingSizeQuantity": basis,
            "servingSizeUnit": "Capsule(s)"}


def ingredient(name, *quantities, children=None, category="botanical"):
    return {"name": name, "quantity": list(quantities), "category": category,
            "nestedRows": children or []}


def test_main_ingredient_criteria():
    cases = [
        # A single listed ingredient requires no mass conversion.
        ([ingredient("A", amount(100, "IU"))], "A", None, "sole_eligible_ingredient"),
        # Aggregate nutrition rows cannot displace a named ingredient.
        ([ingredient("Total Carbohydrates", amount(20, "g")),
          ingredient("A", amount(1))], "A", None, "sole_eligible_ingredient"),
        # Units must be normalized before comparing.
        ([ingredient("A", amount(.1, "g")), ingredient("B", amount(2000, "mcg"))],
         "A", None, "largest_complete_comparable_mass"),
        # Hidden blend constituents prevent confirming the biggest ingredient.
        ([ingredient("Blend", amount(500), category="blend"), ingredient("A", amount(100))],
         None, "A", "largest_among_quantified_only"),
        # A blend's total must not compete against its quantified components.
        ([ingredient("Blend", amount(5000), category="blend", children=[
            ingredient("A", amount(50)), ingredient("B", amount(100))]),
          ingredient("C", amount(200))], "C", None, "largest_complete_comparable_mass"),
        # Incompatible units and NP placeholders are never treated as zero mass.
        ([ingredient("A", amount(200)), ingredient("B", amount(100, "IU"))],
         None, "A", "largest_among_quantified_only"),
        ([ingredient("A", amount(200)), ingredient("B", amount(0, "NP"))],
         None, "A", "largest_among_quantified_only"),
        # Equal masses remain ties; different serving amounts cannot be mixed.
        ([ingredient("A", amount(100)), ingredient("B", amount(.1, "g"))],
         None, None, "unknown_tie_or_serving_disagreement"),
        ([ingredient("A", amount(100, basis=1)), ingredient("B", amount(200, basis=2))],
         None, None, "unknown_tie_or_serving_disagreement"),
        ([ingredient("A", amount(100), amount(200, basis=2)),
          ingredient("B", amount(50), amount(300, basis=2))],
         None, None, "unknown_tie_or_serving_disagreement"),
        # Chemical-form nesting is not silently flattened into blend components.
        ([ingredient("Magnesium", amount(100), children=[
            ingredient("Magnesium citrate", amount(600))])],
         None, None, "unknown_no_comparable_amounts"),
        # Bounds cannot be ranked as if they were exact amounts.
        ([ingredient("A", amount(100)), ingredient("B", amount(200, operator="<"))],
         None, "A", "largest_among_quantified_only"),
        # A conflict on another serving basis cannot be hidden by a complete one.
        ([ingredient("A", amount(100), amount(200, basis=2)), ingredient("B", amount(50))],
         None, "A", "largest_among_quantified_only"),
        # Multiple conflicting amounts for the only quantified row need review.
        ([ingredient("A", amount(100), amount(200)), ingredient("B", amount(0, "NP"))],
         None, None, "unknown_no_comparable_amounts"),
    ]
    labels = [{"id": i, "fullName": f"Case {i}", "ingredientRows": rows}
              for i, (rows, *_) in enumerate(cases)]
    output = runpy.run_path(str(SCRIPT), init_globals={"raw_labels": labels})
    for row, (_, name, partial, status) in zip(output["main_ingredients"], cases):
        assert (row["main_ingredient"], row["largest_among_quantified"], row["status"]) == (
            name, partial, status), row


def test_early_exits_avoid_amount_calculations():
    class UnreadableQuantities(list):
        def __iter__(self):
            raise AssertionError("Sole ingredient must not read quantities")

    sole = ingredient("Sole")
    sole["quantity"] = UnreadableQuantities()
    labels = [
        {"id": 1, "ingredientRows": [sole]},
        {"id": 2, "ingredientRows": [ingredient("Known", amount(3, "g")),
                                     ingredient("Unknown", amount(0, "NP"))]},
    ]
    output = runpy.run_path(str(SCRIPT), init_globals={"raw_labels": labels})
    assert output["main_ingredients"][0]["main_ingredient"] == "Sole"
    assert output["main_ingredients"][1]["largest_among_quantified"] == "Known"
    assert output["work_counts"]["ingredient_mass_conversions"] == 0
    assert output["work_counts"]["groups_requiring_mass_ranking"] == 0
