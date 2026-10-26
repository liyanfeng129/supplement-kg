"""The stores: named label sets, runs created whole, todo read from the rows.

What is pinned: a label set lives under its name with a `passages_sha` in its
manifest; the same passages saved again write nothing, and different passages
under a used name are refused; a bad or duplicated row is refused before
anything is written. A run is named after its label set and the minute, is
created with its ids and manifest together or not at all, and a second run in
the same minute gets the next number. What is left is read from the rows
themselves; a torn trailing line costs one record, not the run, and is only
ever repaired by the store, never by the loader.

An embedding lives inside its label set under its own name: created once with
its manifest, refused for a used name or a missing set; its chunks are
numbered, never visible half-written, and read back in the manifest's id
order; failed ids count as embedded; the set's own files are never touched,
and an embedding of other passages is refused.

A run's profile is written once, beside the run, and read back as written;
it is refused if either of its files exists, and on read it is refused when
the run's rows or manifest are not the ones it was computed from, or when its
labels were written without their manifest.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from engine.repositories.dsld.labels import loader as label_loader
from engine.repositories.dsld.labels import paths as label_paths
from engine.repositories.dsld.labels import store
from engine.repositories.dsld.runs import loader as runs
from engine.repositories.dsld.runs import paths as run_paths
from engine.repositories.dsld.runs import store as run_store
from engine.repositories.dsld.runs.schema import ProfiledLabel


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


# --- embeddings -------------------------------------------------------------


@pytest.fixture
def labels_root(tmp_path, monkeypatch):
    """Label sets under a temp root, with one set `tiny` of labels 1 to 4."""
    root = tmp_path / "labels"
    monkeypatch.setattr(label_paths, "LABELS_ROOT", root)
    store.write_label_set("tiny", _labels(1, 2, 3, 4), {"renderer_version": 4})
    return root


def _embedding_manifest(ids, passages_sha=None):
    sha = passages_sha or _manifest(label_paths.set_dir("tiny"))["passages_sha"]
    return {"passages_sha": sha, "ids": ids, "instruction": "Find the product", "config_hash": "abc"}


def _rows(*ids):
    """Label `i`'s vector is [i, 0, 0]: a row a test can recognise."""
    return np.array(ids, dtype=np.int64), np.array([[i, 0, 0] for i in ids], dtype=np.float32)


def test_an_embedding_is_a_folder_in_its_label_set_with_its_manifest(labels_root):
    d = store.create_embedding("tiny", "first", _embedding_manifest([1, 2]))

    assert d == label_paths.embedding_dir("tiny", "first")
    assert d.parent == labels_root / "tiny" / label_paths.EMBEDDINGS_DIR
    assert label_loader.read_embedding_manifest("tiny", "first")["ids"] == [1, 2]


def test_a_used_embedding_name_is_refused(labels_root):
    store.create_embedding("tiny", "first", _embedding_manifest([1, 2]))

    with pytest.raises(ValueError, match="already exists"):
        store.create_embedding("tiny", "first", _embedding_manifest([3]))

    assert label_loader.read_embedding_manifest("tiny", "first")["ids"] == [1, 2]


def test_an_embedding_of_a_missing_set_is_refused(labels_root):
    with pytest.raises(FileNotFoundError, match="no label set"):
        store.create_embedding("absent", "first", _embedding_manifest([1]))

    assert not (labels_root / "absent").exists()


def test_vectors_come_back_in_the_manifest_s_id_order(labels_root):
    store.create_embedding("tiny", "first", _embedding_manifest([4, 1, 3, 2]))
    store.append_chunk("tiny", "first", *_rows(3, 1))
    store.append_chunk("tiny", "first", *_rows(2, 4))

    ids, vectors = label_loader.load_embedding("tiny", "first")

    assert ids.tolist() == [4, 1, 3, 2]
    assert vectors[:, 0].tolist() == [4.0, 1.0, 3.0, 2.0]
    assert (ids.dtype, vectors.dtype) == (np.int64, np.float32)


def test_a_failed_label_has_no_row_but_counts_as_embedded(labels_root):
    store.create_embedding("tiny", "first", _embedding_manifest([1, 2, 3]))
    store.append_chunk("tiny", "first", *_rows(1))
    store.append_failed("tiny", "first", [{"dsld_id": 2, "failed": "error"}])

    assert label_loader.embedded_ids("tiny", "first") == {1, 2}
    assert label_loader.load_embedding("tiny", "first")[0].tolist() == [1]


def test_a_chunk_is_never_visible_half_written(labels_root, monkeypatch):
    store.create_embedding("tiny", "first", _embedding_manifest([1, 2]))
    store.append_chunk("tiny", "first", *_rows(1))

    def crash_mid_write(file, **arrays):
        file.write(b"PK\x03\x04 half a zip")
        raise OSError("disk full")

    with monkeypatch.context() as patch, pytest.raises(OSError):
        patch.setattr(np, "savez", crash_mid_write)
        store.append_chunk("tiny", "first", *_rows(2))

    assert label_loader.embedded_ids("tiny", "first") == {1}
    assert label_loader.load_embedding("tiny", "first")[0].tolist() == [1]


def test_the_label_set_s_own_files_are_never_touched(labels_root):
    d = label_paths.set_dir("tiny")
    before = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}

    store.create_embedding("tiny", "first", _embedding_manifest([1, 2]))
    store.append_chunk("tiny", "first", *_rows(1))
    store.append_failed("tiny", "first", [{"dsld_id": 2, "failed": "empty"}])

    assert {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()} == before


def test_an_embedding_of_other_passages_is_refused(labels_root):
    store.create_embedding("tiny", "stale", _embedding_manifest([1], passages_sha="0000000000000000"))
    store.append_chunk("tiny", "stale", *_rows(1))

    with pytest.raises(ValueError, match="passages_sha"):
        label_loader.load_embedding("tiny", "stale")


# --- profiles ---------------------------------------------------------------


_PROFILED = [
    ProfiledLabel(1, "both", ["verbatim", "stitched"], ["invented"]),
    ProfiledLabel(2, "neither", [], []),
]


@pytest.fixture
def profiled_run(tmp_path):
    """A run with two rows and a manifest, not yet profiled."""
    d = tmp_path / "tiny-2610220000-0"
    _write(d, [{"dsld_id": 1, "status": "ok"}, {"dsld_id": 2, "status": "ok"}])
    run_paths.manifest_path(d).parent.mkdir(parents=True)
    run_paths.manifest_path(d).write_text(json.dumps({"config_hash": "abc", "passages_sha": "def"}))
    return d


def _write_profile(d):
    run_store.write_profile(
        d, _PROFILED, {"run": d.name, "config_hash": "abc", "passages_sha": "def", "classifier_version": 1}
    )


def test_a_profile_reads_back_as_written_with_the_rows_sha(profiled_run):
    _write_profile(profiled_run)

    manifest, labels = runs.read_profile(profiled_run)
    assert labels == _PROFILED
    assert manifest["classifier_version"] == 1
    assert len(manifest["rows_sha"]) == 16


def test_a_second_profile_is_refused(profiled_run):
    _write_profile(profiled_run)

    with pytest.raises(FileExistsError):
        _write_profile(profiled_run)


def test_profile_labels_without_their_manifest_are_refused_on_write_and_read(profiled_run):
    _write_profile(profiled_run)
    run_paths.manifest_path(profiled_run, run_paths.PROFILE_MANIFEST_NAME).unlink()

    with pytest.raises(ValueError, match="crashed write"):
        runs.read_profile(profiled_run)
    with pytest.raises(FileExistsError):
        _write_profile(profiled_run)


def test_a_run_without_a_profile_raises_on_read(profiled_run):
    with pytest.raises(FileNotFoundError):
        runs.read_profile(profiled_run)


def test_rows_changed_in_place_make_the_profile_stale(profiled_run):
    _write_profile(profiled_run)
    _write(profiled_run, [{"dsld_id": 1, "status": "ok"}, {"dsld_id": 2, "status": "error"}])

    with pytest.raises(ValueError, match="rows_sha"):
        runs.read_profile(profiled_run)


@pytest.mark.parametrize("field", ["config_hash", "passages_sha"])
def test_another_run_manifest_makes_the_profile_stale(profiled_run, field):
    _write_profile(profiled_run)
    m = json.loads(run_paths.manifest_path(profiled_run).read_text())
    m[field] = "other"
    run_paths.manifest_path(profiled_run).write_text(json.dumps(m))

    with pytest.raises(ValueError, match=field):
        runs.read_profile(profiled_run)


def test_a_profile_write_leaves_no_temporary_files(profiled_run):
    _write_profile(profiled_run)

    hidden = [p.name for p in profiled_run.rglob(".*")]
    assert hidden == []
