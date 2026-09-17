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


# --- the singleton ----------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_loader():
    """Every test gets a cold loader.

    The loader caches parsed corpora on one shared instance, so a test that
    read a fixture would otherwise hand its corpus to the next test.
    """
    loader.reset()
    yield
    loader.reset()


def test_there_is_only_one_loader():
    assert loader.loader() is loader.loader() is loader.DsldLoader()


def test_the_corpus_is_read_once_and_shared(records, monkeypatch):
    """The point of the singleton: two callers asking for the same file must
    not each pay for it, and must not be able to see two different versions."""
    loader.load_flattened(records)
    records.write_text("")  # a second read of the file would now yield nothing
    assert len(loader.load_flattened(records)) == 3


def test_reset_drops_the_cache(records):
    loader.load_flattened(records)
    loader.reset()
    records.write_text(json.dumps(RECORDS[0]) + "\n")
    assert len(loader.load_flattened(records)) == 1


def test_a_returned_list_can_be_mutated_without_damaging_the_cache(records):
    """The cached list is shared; handing it out directly would let one caller
    sort or clear the corpus for everyone after it."""
    loader.load_flattened(records).clear()
    assert len(loader.load_flattened(records)) == 3


def test_different_paths_are_cached_separately(records, tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps(RECORDS[0]) + "\n")
    assert len(loader.load_flattened(records)) == 3
    assert len(loader.load_flattened(other)) == 1


# --- load_flattened ---------------------------------------------------------


def test_load_flattened_reads_every_record_in_file_order(records):
    assert [lab.dsld_id for lab in loader.load_flattened(records)] == [709, 885, 889]


def test_read_all_still_works_for_existing_callers(records):
    """The notebook and CLI call it; the alias keeps them running."""
    assert loader.read_all(records) == loader.load_flattened(records)


# --- load_raw ---------------------------------------------------------------


def _dump_with(root, dsld_id, doc):
    lo = (int(dsld_id) // loader.DUMP_BUCKET_SIZE) * loader.DUMP_BUCKET_SIZE
    bucket = root / f"{lo:06d}-{lo + loader.DUMP_BUCKET_SIZE - 1:06d}"
    bucket.mkdir(parents=True, exist_ok=True)
    (bucket / f"{dsld_id}.json").write_text(json.dumps(doc))
    return root


def test_load_raw_returns_the_whole_published_record(tmp_path):
    """Every field, including ones this repo does not consume yet -- keeping
    those is the reason a dump exists at all."""
    doc = {"id": 709, "fullName": "Huperzine A", "someFieldWeIgnore": [1, 2]}
    root = _dump_with(tmp_path / "dump", 709, doc)
    assert loader.load_raw(709, root) == doc


def test_load_raw_finds_the_bucket_for_an_id(tmp_path):
    root = _dump_with(tmp_path / "dump", 18162, {"id": 18162})
    assert loader.dump_path(18162, root).parent.name == "018000-018999"
    assert loader.load_raw("18162", root)["id"] == 18162


def test_load_raw_on_an_absent_id_raises_dsld_not_found(tmp_path):
    """Matches what the API client raises for an id it does not know, so a
    caller can swap a dump for the API without changing its handling."""
    with pytest.raises(loader.DSLDNotFound, match="99999"):
        loader.load_raw(99999, tmp_path / "dump")


# --- load_gold --------------------------------------------------------------


GOLD = [
    {
        "dsld_id": 709,
        "name": "Huperzine A",
        "purpose": [{"text": "supports memory", "evidence": "Supports Memory*", "grounded": True}],
        "risk": [{"text": "do not exceed intake", "evidence": "WARNING", "grounded": True}],
        "model": "claude-opus-5",
        "extracted_at": "2026-09-08T13:19:16Z",
    },
    {"dsld_id": 885, "name": "Spirulina", "purpose": [], "risk": []},
]


@pytest.fixture
def gold(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in GOLD))
    return p


def test_load_gold_reads_claims_with_their_evidence(gold):
    """A claim that cannot be pointed back at the passage cannot be audited,
    and the gold set is a reference reading, not ground truth."""
    rows = loader.load_gold(gold)
    assert [g.dsld_id for g in rows] == [709, 885]
    assert rows[0].purpose[0].evidence == "Supports Memory*"
    assert rows[0].purpose[0].grounded is True
    assert rows[0].model == "claude-opus-5"


def test_a_gold_row_with_no_claims_is_valid(gold):
    """Most labels declare no risk; an empty list is an answer, not a failure."""
    assert loader.load_gold(gold)[1].purpose == []


def test_gold_by_id_is_keyed_for_the_join(gold):
    by_id = loader.gold_by_id(gold)
    assert set(by_id) == {709, 885}
    assert by_id[709].name == "Huperzine A"


def test_a_missing_gold_set_says_how_to_rebuild_it(tmp_path):
    """It is committed outside data/ and costs money to regenerate, so the
    message has to say that rather than just naming a path."""
    with pytest.raises(FileNotFoundError, match="gold_set"):
        loader.load_gold(tmp_path / "absent.jsonl")


def test_a_malformed_gold_line_names_its_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(GOLD[0]) + "\nnot json\n")
    with pytest.raises(ValueError, match=":2:"):
        loader.load_gold(p)
