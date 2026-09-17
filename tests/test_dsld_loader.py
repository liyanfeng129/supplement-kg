"""The DSLD loader: what it returns, and what it says about what it left out.

The old loader returned a bare list and printed its counts, so a caller could
not tell a short result from a filtered one. These tests pin the thing that
replaced it: every record dropped is accounted for on the `Selection`, in a
field rather than on stdout.

No GPU, no network, no model. The loader reads files and filters.
"""

from __future__ import annotations

import json

import pytest

from engine.repositories.dsld import loader, store
from engine.repositories.dsld.schema import Label

RECORDS = [
    {"dsld_id": 709, "name": "Huperzine A", "passage": "PRODUCT: Huperzine A"},
    {"dsld_id": 885, "name": "Spirulina", "passage": "PRODUCT: Spirulina"},
    {"dsld_id": 889, "name": "Goldenseal", "passage": "PRODUCT: Goldenseal"},
]


@pytest.fixture
def records(tmp_path):
    p = tmp_path / "labels.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    return p


# --- schema -----------------------------------------------------------------


def test_label_coerces_a_string_id():
    """Mixed id types silently break every membership test resume depends on."""
    assert Label.from_dict({"dsld_id": "709", "name": "x", "passage": "p"}).dsld_id == 709


def test_label_without_a_passage_is_refused():
    """A label with no text is not an extractable label; failing here beats
    failing per-record deep inside a batch."""
    with pytest.raises(ValueError):
        Label.from_dict({"dsld_id": 1, "name": "x"})


def test_label_tolerates_a_missing_name():
    """Name is a convenience for reading rows, not something a run depends on."""
    assert Label.from_dict({"dsld_id": 1, "passage": "p"}).name == ""


def test_label_is_frozen():
    """A label is input: mutating it would leave the manifest describing text
    that no longer exists."""
    lab = Label.from_dict(RECORDS[0])
    with pytest.raises(Exception):
        lab.passage = "changed"  # type: ignore[misc]


# --- reading ----------------------------------------------------------------


def test_reads_every_record_in_file_order(records):
    assert [lab.dsld_id for lab in loader.read_all(records)] == [709, 885, 889]


def test_missing_source_says_where_and_why(tmp_path):
    """raw/ is gitignored, so a fresh clone hits this -- the message has to
    explain that rather than just naming a path."""
    with pytest.raises(FileNotFoundError, match="gitignored"):
        loader.read_all(tmp_path / "absent.jsonl")


def test_a_malformed_line_names_its_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(RECORDS[0]) + "\nnot json\n")
    with pytest.raises(ValueError, match=":2:"):
        loader.read_all(p)


def test_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "gaps.jsonl"
    p.write_text(json.dumps(RECORDS[0]) + "\n\n" + json.dumps(RECORDS[1]) + "\n")
    assert len(loader.read_all(p)) == 2


# --- selection --------------------------------------------------------------


def test_load_reports_what_is_available(records):
    sel = loader.load(records)
    assert len(sel) == 3 and sel.n_available == 3
    assert sel.n_already_done == 0 and sel.missing_ids == []


def test_selection_is_iterable_and_carries_its_ids(records):
    sel = loader.load(records)
    assert [lab.dsld_id for lab in sel] == sel.ids == [709, 885, 889]


def test_ids_select_and_preserve_caller_order(records):
    """An explicit id list is usually a priority order; re-sorting hides that
    it was ignored."""
    assert loader.load(records, ids=[889, 709]).ids == [889, 709]


def test_unknown_ids_are_reported_not_silently_dropped(records):
    """A typo'd id and a genuinely absent one look identical in a bare result."""
    sel = loader.load(records, ids=[709, 99999])
    assert sel.ids == [709]
    assert sel.missing_ids == [99999]


def test_limit_caps_the_result(records):
    assert len(loader.load(records, limit=2)) == 2


# --- resume -----------------------------------------------------------------


def _run_with(tmp_path, done_ids):
    d = store.run_dir(root=tmp_path / "runs")
    with store.rows_path(d).open("w") as fh:
        for i in done_ids:
            fh.write(json.dumps({"dsld_id": i, "status": "ok"}) + "\n")
    return d


def test_resume_drops_done_labels_and_counts_them(records, tmp_path):
    d = _run_with(tmp_path, [709, 885])
    sel = loader.load(records, resume_from=d)
    assert sel.ids == [889]
    assert sel.n_already_done == 2
    assert sel.n_available == 3


def test_resume_against_an_empty_run_changes_nothing(records, tmp_path):
    d = _run_with(tmp_path, [])
    sel = loader.load(records, resume_from=d)
    assert len(sel) == 3 and sel.n_already_done == 0


def test_limit_applies_after_resume(records, tmp_path):
    """`--limit 10` must mean ten labels that will be processed.

    Applying it before the resume filter let a resumed run hand back nothing
    and look like completion.
    """
    d = _run_with(tmp_path, [709, 885])
    sel = loader.load(records, limit=1, resume_from=d)
    assert sel.ids == [889]


def test_resume_and_ids_compose(records, tmp_path):
    d = _run_with(tmp_path, [709])
    sel = loader.load(records, ids=[709, 889], resume_from=d)
    assert sel.ids == [889] and sel.n_already_done == 1


# --- the summary line -------------------------------------------------------


def test_summary_mentions_each_kind_of_exclusion(records, tmp_path):
    d = _run_with(tmp_path, [709])
    summary = loader.load(records, ids=[709, 885, 99999], resume_from=d).summary()
    assert "1 to do" in summary
    assert "already done" in summary
    assert "99999" in summary


def test_summary_stays_quiet_when_nothing_was_excluded(records):
    summary = loader.load(records).summary()
    assert summary == "3 to do of 3 available"
