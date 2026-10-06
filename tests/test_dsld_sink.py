"""The sink: rows reach the run's file in batches, and a crash loses only the buffer.

What is pinned: `add` buffers and flushes at `flush_every` rows or after
`flush_seconds`; a flush appends and never rewrites; `close` flushes what is
left, including when the block is left by an exception; and opening a sink on
a run with a torn trailing line repairs it first, so the next append does not
strand that line mid-file.

No model, no GPU. The sink writes files.
"""

from __future__ import annotations

import json

import pytest

from engine import config
from engine.repositories.dsld.runs import loader, paths, store


def rows_on_disk(run) -> list[dict]:
    return loader.read_rows(run)


def test_rows_wait_in_the_buffer_until_flush_every(tmp_path):
    run = tmp_path / "run"
    with store.Sink(run, flush_every=3, flush_seconds=3600) as sink:
        sink.add({"dsld_id": 1})
        sink.add({"dsld_id": 2})
        assert rows_on_disk(run) == []

        sink.add({"dsld_id": 3})
        assert [r["dsld_id"] for r in rows_on_disk(run)] == [1, 2, 3]


def test_a_quiet_buffer_is_flushed_after_flush_seconds(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(store.time, "monotonic", lambda: clock[0])
    run = tmp_path / "run"
    with store.Sink(run, flush_every=1000, flush_seconds=30) as sink:
        sink.add({"dsld_id": 1})
        assert rows_on_disk(run) == []

        clock[0] = 31.0
        sink.add({"dsld_id": 2})
        assert [r["dsld_id"] for r in rows_on_disk(run)] == [1, 2]


def test_close_flushes_what_is_left(tmp_path):
    run = tmp_path / "run"
    with store.Sink(run, flush_every=64) as sink:
        sink.add({"dsld_id": 1})

    assert [r["dsld_id"] for r in rows_on_disk(run)] == [1]


def test_an_exception_still_flushes_what_finished(tmp_path):
    """Ctrl-C mid-run: every finished row reaches disk before the error leaves."""
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        with store.Sink(run, flush_every=64) as sink:
            sink.add({"dsld_id": 1})
            raise KeyboardInterrupt

    assert [r["dsld_id"] for r in rows_on_disk(run)] == [1]


def test_a_flush_appends_to_an_existing_run(tmp_path):
    run = tmp_path / "run"
    with store.Sink(run, flush_every=1) as sink:
        sink.add({"dsld_id": 1})
    with store.Sink(run, flush_every=1) as sink:
        sink.add({"dsld_id": 2})

    assert [r["dsld_id"] for r in rows_on_disk(run)] == [1, 2]


def test_opening_on_a_torn_run_repairs_it_before_appending(tmp_path):
    """Appending after a torn line would strand it mid-file, where it is fatal."""
    run = tmp_path / "run"
    paths.rows_path(run).parent.mkdir(parents=True)
    with paths.rows_path(run).open("w") as fh:
        fh.write(json.dumps({"dsld_id": 1}) + "\n")
        fh.write('{"dsld_id": 2, "sta')  # killed mid-flush

    with store.Sink(run, flush_every=1) as sink:
        sink.add({"dsld_id": 3})

    assert [r["dsld_id"] for r in rows_on_disk(run)] == [1, 3]


def test_a_sink_on_a_new_run_creates_the_rows_file(tmp_path):
    run = tmp_path / "fresh"
    with store.Sink(run, flush_every=1) as sink:
        sink.add({"dsld_id": 1})

    assert [r["dsld_id"] for r in rows_on_disk(run)] == [1]


# --- settings from configs/engine.toml --------------------------------------


def test_flush_settings_come_from_the_committed_file(tmp_path):
    sink = store.Sink(tmp_path / "run")

    assert (sink.flush_every, sink.flush_seconds) == (64, 30.0)


def test_flush_settings_follow_the_file(monkeypatch, tmp_path):
    file = tmp_path / "engine.toml"
    file.write_text("[store]\nflush_every = 5\nflush_seconds = 2.5\n")
    monkeypatch.setattr(config, "ENGINE_CONFIG_PATH", file)

    sink = store.Sink(tmp_path / "run")

    assert (sink.flush_every, sink.flush_seconds) == (5, 2.5)


def test_constructor_arguments_beat_the_file(monkeypatch, tmp_path):
    file = tmp_path / "engine.toml"
    file.write_text("[store]\nflush_every = 5\nflush_seconds = 2.5\n")
    monkeypatch.setattr(config, "ENGINE_CONFIG_PATH", file)

    sink = store.Sink(tmp_path / "run", flush_every=1, flush_seconds=0)

    assert (sink.flush_every, sink.flush_seconds) == (1, 0)


def test_a_missing_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENGINE_CONFIG_PATH", tmp_path / "absent.toml")

    sink = store.Sink(tmp_path / "run")

    assert (sink.flush_every, sink.flush_seconds) == (64, 30.0)
