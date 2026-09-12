"""Resume is computed from the data, so the data must survive a crash.

The run loop itself is not a seam -- it is the GPU and IO path, covered by
`smoke` -- but resume-by-absence and the torn-line rule are pure file
behaviour, and getting them wrong silently loses or corrupts a multi-day run.
"""

from __future__ import annotations

import json

from engine.pipeline import store


def _write(d, rows):
    with store.rows_path(d).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_done_ids_are_read_from_the_rows_themselves(tmp_path):
    d = store.run_dir("20260101T000000Z", root=tmp_path)
    _write(d, [{"dsld_id": 1, "status": "ok"}, {"dsld_id": 2, "status": "empty"}])

    assert store.done_ids(d) == {1, 2}


def test_a_torn_trailing_line_is_discarded_not_fatal(tmp_path):
    """A crash mid-write costs one record rather than corrupting the run."""
    d = store.run_dir("20260101T000000Z", root=tmp_path)
    with store.rows_path(d).open("w") as fh:
        fh.write(json.dumps({"dsld_id": 1, "status": "ok"}) + "\n")
        fh.write('{"dsld_id": 2, "sta')  # killed mid-flush

    assert store.done_ids(d) == {1}
    assert len(store.read_rows(d)) == 1


def test_repair_rewrites_the_file_without_the_torn_line(tmp_path):
    """Appending after a torn line would strand it mid-file, where it is fatal."""
    d = store.run_dir("20260101T000000Z", root=tmp_path)
    with store.rows_path(d).open("w") as fh:
        fh.write(json.dumps({"dsld_id": 1, "status": "ok"}) + "\n")
        fh.write('{"dsld_id": 2, "sta')

    assert store.repair(d) == 1

    with store.rows_path(d).open("a") as fh:
        fh.write(json.dumps({"dsld_id": 3, "status": "ok"}) + "\n")

    assert store.done_ids(d) == {1, 3}


def test_an_empty_run_dir_has_no_done_ids(tmp_path):
    d = store.run_dir("20260101T000000Z", root=tmp_path)

    assert store.done_ids(d) == set()
    assert store.read_rows(d) == []


def test_manifest_round_trips(tmp_path):
    d = store.run_dir("20260101T000000Z", root=tmp_path)
    store.write_manifest(d, {"run_id": d.name, "note": "tightened boilerplate rule"})

    assert store.read_manifest(d)["note"] == "tightened boilerplate rule"


def test_run_dirs_follow_the_manifest_and_raw_convention(tmp_path):
    """Manifests are committed; raw payloads are gitignored."""
    d = store.run_dir("20260101T000000Z", root=tmp_path)

    assert (d / "raw").is_dir()
    assert (d / "manifests").is_dir()
    assert store.rows_path(d).parent.name == "raw"
    assert store.manifest_path(d).parent.name == "manifests"
