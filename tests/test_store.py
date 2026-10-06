"""The store: named label sets, runs created whole, done read from the rows.

What is pinned: a label set lives under its name with a `passages_sha` in its
manifest; the same passages saved again write nothing, and different passages
under a used name are refused; a bad or duplicated row is refused before
anything is written. A run is named after its label set and the minute, is
created with its ids and manifest together or not at all, and a second run in
the same minute gets the next number. Done is read from the rows themselves,
and a torn trailing line costs one record, not the run.
"""

from __future__ import annotations

import json

import pytest

from engine.repositories.dsld import loader, store


def _write(d, rows):
    store.rows_path(d).parent.mkdir(parents=True, exist_ok=True)
    with store.rows_path(d).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _labels(*ids):
    return [{"dsld_id": i, "name": f"label {i}", "passage": f"passage {i}"} for i in ids]


# --- label sets -------------------------------------------------------------


def _manifest(d):
    return json.loads((d / store.LABEL_SET_MANIFEST_NAME).read_text())


def test_a_label_set_is_written_under_its_name_with_its_passages_sha(tmp_path):
    sha = store.write_label_set("tiny-r2", _labels(1, 2), {"renderer_version": 2}, root=tmp_path)

    d = tmp_path / "tiny-r2"
    rows = [json.loads(l) for l in (d / store.LABEL_SET_ROWS_NAME).read_text().splitlines()]
    assert rows == _labels(1, 2)
    assert _manifest(d) == {"name": "tiny-r2", "renderer_version": 2, "passages_sha": sha}
    assert len(sha) == 16


def test_the_same_passages_saved_again_write_nothing(tmp_path):
    first = store.write_label_set("tiny", _labels(1, 2), {"note": "first"}, root=tmp_path)
    second = store.write_label_set("tiny", _labels(1, 2), {"note": "second"}, root=tmp_path)

    assert first == second
    assert _manifest(tmp_path / "tiny")["note"] == "first"  # the existing set is not rewritten


def test_changed_passages_under_a_used_name_are_refused(tmp_path):
    store.write_label_set("tiny", _labels(1, 2), {}, root=tmp_path)
    changed = _labels(1, 2)
    changed[1]["passage"] = "passage 2, edited"

    with pytest.raises(ValueError, match="new name"):
        store.write_label_set("tiny", changed, {}, root=tmp_path)


def test_the_manifest_does_not_change_the_passages_sha(tmp_path):
    a = store.write_label_set("a", _labels(1), {"note": "a"}, root=tmp_path)
    b = store.write_label_set("b", _labels(1), {"note": "b"}, root=tmp_path)

    assert a == b


def test_a_duplicate_id_is_refused_and_nothing_is_written(tmp_path):
    with pytest.raises(ValueError, match="twice"):
        store.write_label_set("dup", _labels(1, 2, 1), {}, root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_row_without_a_passage_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a usable label"):
        store.write_label_set("bad", [{"dsld_id": 1, "name": "x"}], {}, root=tmp_path)


@pytest.mark.parametrize("name", ["", "a/b", ".hidden"])
def test_a_name_that_is_not_a_plain_directory_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match="cannot name"):
        store.write_label_set(name, _labels(1), {}, root=tmp_path)


# --- runs -------------------------------------------------------------------


def test_a_run_is_named_after_its_label_set_and_the_minute(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "run_id", lambda: "2609301512")

    d = store.new_run("gold1000-r1", [889, 709, 885], {"label_set": "gold1000-r1"}, root=tmp_path)

    assert d.name == "gold1000-r1-2609301512-0"
    assert loader.read_ids(d) == [889, 709, 885]
    assert loader.read_manifest(d) == {"label_set": "gold1000-r1"}
    assert store.rows_path(d).parent.name == "raw"
    assert store.manifest_path(d).parent.name == "manifests"


def test_runs_in_the_same_minute_are_numbered_from_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "run_id", lambda: "2609301512")

    names = [store.new_run("s", [i], {}, root=tmp_path).name for i in range(3)]

    assert names == ["s-2609301512-0", "s-2609301512-1", "s-2609301512-2"]


def test_the_time_is_the_utc_minute():
    assert len(store.run_id()) == 10 and store.run_id().isdigit()


def test_a_failed_creation_leaves_no_run_behind(tmp_path, monkeypatch):
    """A crash while creating a run leaves nothing a resume could mistake for one."""
    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "manifest_path", boom)

    with pytest.raises(KeyboardInterrupt):
        store.new_run("s", [1, 2], {}, root=tmp_path)

    assert list(tmp_path.iterdir()) == []


# --- done is read from the rows ---------------------------------------------


def test_done_ids_are_read_from_the_rows_themselves(tmp_path):
    _write(tmp_path, [{"dsld_id": 1, "status": "ok"}, {"dsld_id": 2, "status": "error"}])

    assert loader.done_ids(tmp_path) == {1, 2}


def test_a_torn_trailing_line_is_discarded_not_fatal(tmp_path):
    """A crash mid-write costs one record rather than corrupting the run."""
    _write(tmp_path, [{"dsld_id": 1, "status": "ok"}])
    with store.rows_path(tmp_path).open("a") as fh:
        fh.write('{"dsld_id": 2, "sta')  # killed mid-flush

    assert loader.done_ids(tmp_path) == {1}
    assert len(loader.read_rows(tmp_path)) == 1


def test_repair_rewrites_the_file_without_the_torn_line(tmp_path):
    """Appending after a torn line would strand it mid-file, where it is fatal."""
    _write(tmp_path, [{"dsld_id": 1, "status": "ok"}])
    with store.rows_path(tmp_path).open("a") as fh:
        fh.write('{"dsld_id": 2, "sta')

    assert store.repair(tmp_path) == 1

    with store.rows_path(tmp_path).open("a") as fh:
        fh.write(json.dumps({"dsld_id": 3, "status": "ok"}) + "\n")

    assert loader.done_ids(tmp_path) == {1, 3}


def test_a_run_without_rows_has_no_done_ids(tmp_path):
    d = store.new_run("s", [1], {}, root=tmp_path)

    assert loader.done_ids(d) == set()
    assert loader.read_rows(d) == []
