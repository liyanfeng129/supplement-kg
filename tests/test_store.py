"""The stores: named label sets, runs created whole, todo read from the rows.

What is pinned: a label set lives under its name with a `passages_sha` in its
manifest; the same passages saved again write nothing, and different passages
under a used name are refused; a bad or duplicated row is refused before
anything is written. A run is named after its label set and the minute, is
created with its ids and manifest together or not at all, and a second run in
the same minute gets the next number. What is left is read from the rows
themselves; a torn trailing line costs one record, not the run, and is only
ever repaired by the store, never by the loader.
"""

from __future__ import annotations

import json

import pytest

from engine.repositories.dsld.labels import paths as label_paths
from engine.repositories.dsld.labels import store
from engine.repositories.dsld.runs import loader as runs
from engine.repositories.dsld.runs import paths as run_paths
from engine.repositories.dsld.runs import store as run_store


def _write(d, rows):
    run_paths.rows_path(d).parent.mkdir(parents=True, exist_ok=True)
    with run_paths.rows_path(d).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _labels(*ids):
    return [{"dsld_id": i, "name": f"label {i}", "passage": f"passage {i}"} for i in ids]


# --- label sets -------------------------------------------------------------


def _manifest(d):
    return json.loads((d / label_paths.MANIFEST_NAME).read_text())


def test_a_label_set_is_written_under_its_name_with_its_passages_sha(tmp_path):
    sha = store.write_label_set("tiny-r2", _labels(1, 2), {"renderer_version": 2}, root=tmp_path)

    d = tmp_path / "tiny-r2"
    rows = [json.loads(l) for l in (d / label_paths.PASSAGES_NAME).read_text().splitlines()]
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


# --- runs ---------------------------------------------------------------------


def test_a_run_is_named_after_its_label_set_and_the_minute(tmp_path, monkeypatch):
    monkeypatch.setattr(run_paths, "run_id", lambda: "2609301512")

    d = run_store.new_run("gold1000-r1", [889, 709, 885], {"label_set": "gold1000-r1"}, root=tmp_path)

    assert d.name == "gold1000-r1-2609301512-0"
    assert runs.read_ids(d) == [889, 709, 885]
    assert runs.read_manifest(d) == {"label_set": "gold1000-r1"}
    assert run_paths.rows_path(d).parent.name == "raw"
    assert run_paths.manifest_path(d).parent.name == "manifests"


def test_runs_in_the_same_minute_are_numbered_from_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(run_paths, "run_id", lambda: "2609301512")

    names = [run_store.new_run("s", [i], {}, root=tmp_path).name for i in range(3)]

    assert names == ["s-2609301512-0", "s-2609301512-1", "s-2609301512-2"]


def test_the_time_is_the_utc_minute():
    assert len(run_paths.run_id()) == 10 and run_paths.run_id().isdigit()


def test_runs_are_redirected_by_patching_their_paths_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(run_paths, "RUNS_ROOT", tmp_path / "runs")

    d = run_store.new_run("s", [1], {})

    assert d.parent == tmp_path / "runs"


def test_a_failed_creation_leaves_no_run_behind(tmp_path, monkeypatch):
    """A crash while creating a run leaves nothing a resume could mistake for one."""
    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_paths, "manifest_path", boom)

    with pytest.raises(KeyboardInterrupt):
        run_store.new_run("s", [1, 2], {}, root=tmp_path)

    assert list(tmp_path.iterdir()) == []


# --- todo is read from the rows ---------------------------------------------


def _run(tmp_path, ids):
    return run_store.new_run("s", ids, {}, root=tmp_path)


def test_todo_is_the_run_ids_without_a_row_in_run_order(tmp_path):
    d = _run(tmp_path, [5, 2, 4, 1])
    _write(d, [{"dsld_id": 4, "status": "ok"}, {"dsld_id": 5, "status": "error"}])

    assert runs.todo_ids(d) == [2, 1]


def test_a_run_without_rows_has_everything_to_do(tmp_path):
    d = _run(tmp_path, [1, 2])

    assert runs.todo_ids(d) == [1, 2]
    assert runs.read_rows(d) == []


def test_a_torn_trailing_line_is_todo_not_fatal(tmp_path):
    """A crash mid-write costs one record rather than corrupting the run."""
    d = _run(tmp_path, [1, 2])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    with run_paths.rows_path(d).open("a") as fh:
        fh.write('{"dsld_id": 2, "sta')  # killed mid-flush

    assert runs.todo_ids(d) == [2]
    assert runs.read_rows(d) == [{"dsld_id": 1, "status": "ok"}]


def test_a_write_torn_inside_a_character_is_todo_not_fatal(tmp_path):
    """Rows keep non-ASCII text, so a crash can split a character's bytes."""
    d = _run(tmp_path, [1, 2])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    with run_paths.rows_path(d).open("ab") as fh:
        fh.write('{"dsld_id": 2, "raw": "\u00b5g'.encode()[:-2])  # half of "µ"

    assert runs.todo_ids(d) == [2]


def test_reading_a_torn_run_leaves_the_file_as_it_is(tmp_path):
    """The loader never repairs: mid-run, the tail may be a row still being written."""
    d = _run(tmp_path, [1, 2])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    with run_paths.rows_path(d).open("a") as fh:
        fh.write('{"dsld_id": 2, "sta')
    before = run_paths.rows_path(d).read_bytes()

    runs.read_rows(d)
    runs.todo_ids(d)

    assert run_paths.rows_path(d).read_bytes() == before


def test_a_whole_last_row_without_its_newline_is_done(tmp_path):
    d = _run(tmp_path, [1, 2])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    with run_paths.rows_path(d).open("a") as fh:
        fh.write(json.dumps({"dsld_id": 2, "status": "ok"}))

    assert runs.todo_ids(d) == []


def test_a_broken_line_in_the_middle_raises(tmp_path):
    """Only the last line can be torn; anywhere else it is corruption."""
    d = _run(tmp_path, [1, 2, 3])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    with run_paths.rows_path(d).open("a") as fh:
        fh.write('{"dsld_id": 2, "sta\n')
        fh.write(json.dumps({"dsld_id": 3, "status": "ok"}) + "\n")

    with pytest.raises(ValueError, match=":2:"):
        runs.read_rows(d)
    with pytest.raises(ValueError):
        runs.todo_ids(d)


# --- repair -----------------------------------------------------------------


def test_repair_cuts_a_torn_tail_so_the_next_row_can_be_appended(tmp_path):
    """Appending after a torn line would strand it mid-file, where it is fatal."""
    d = _run(tmp_path, [1, 2, 3])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    with run_paths.rows_path(d).open("a") as fh:
        fh.write('{"dsld_id": 2, "sta')

    run_store.repair(d)
    with run_paths.rows_path(d).open("a") as fh:
        fh.write(json.dumps({"dsld_id": 3, "status": "ok"}) + "\n")

    assert runs.todo_ids(d) == [2]
    assert len(runs.read_rows(d)) == 2


def test_repair_keeps_a_whole_row_that_lost_its_newline(tmp_path):
    """The loader counted it done, so cutting it would leave its id with no row."""
    d = _run(tmp_path, [1, 2, 3])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    with run_paths.rows_path(d).open("a") as fh:
        fh.write(json.dumps({"dsld_id": 2, "status": "ok"}))

    run_store.repair(d)
    with run_paths.rows_path(d).open("a") as fh:
        fh.write(json.dumps({"dsld_id": 3, "status": "ok"}) + "\n")

    assert runs.todo_ids(d) == []


def test_repair_cuts_a_tail_longer_than_one_read(tmp_path):
    d = _run(tmp_path, [1, 2])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    with run_paths.rows_path(d).open("a") as fh:
        fh.write('{"dsld_id": 2, "raw": "' + "x" * 200_000)

    run_store.repair(d)

    assert run_paths.rows_path(d).read_text() == json.dumps({"dsld_id": 1, "status": "ok"}) + "\n"


def test_repair_of_a_lone_torn_row_empties_the_file(tmp_path):
    d = _run(tmp_path, [1])
    run_paths.rows_path(d).write_text('{"dsld_id": 1, "sta')

    run_store.repair(d)

    assert run_paths.rows_path(d).read_text() == ""


def test_repair_leaves_a_whole_file_alone(tmp_path):
    d = _run(tmp_path, [1])
    _write(d, [{"dsld_id": 1, "status": "ok"}])
    before = run_paths.rows_path(d).read_bytes()

    run_store.repair(d)

    assert run_paths.rows_path(d).read_bytes() == before


def test_repair_of_a_run_without_rows_creates_nothing(tmp_path):
    d = _run(tmp_path, [1])

    run_store.repair(d)

    assert not run_paths.rows_path(d).exists()
