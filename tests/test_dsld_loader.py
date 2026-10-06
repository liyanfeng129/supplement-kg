"""The DSLD loader: the raw dump for cleaning, whole label sets for the LLM.

What is pinned: `load_raw` returns one record, several, or a draw from the
dump, sorted unless told otherwise and never randomly without a seed.
`load_flattened` reads a label set by name, returns its labels keyed by id
with the set's `passages_sha`, refuses a set edited in place, and parses each
set once.

No GPU, no network, no model. The loader reads files.
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


@pytest.fixture(autouse=True)
def _fresh_loader():
    """Every test gets a cold loader, so one test's cached set is not the next's."""
    loader.reset()
    yield
    loader.reset()


@pytest.fixture
def labels_root(tmp_path, monkeypatch):
    root = tmp_path / "labels"
    monkeypatch.setattr(store, "LABELS_ROOT", root)
    monkeypatch.setattr(loader, "LABELS_ROOT", root)
    return root


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



# --- load_flattened: a whole label set ------------------------------------


def test_a_set_is_read_by_its_name(labels_root):
    sha = store.write_label_set("tiny", RECORDS, {}, root=labels_root)

    got_sha, labels = loader.load_flattened("tiny")

    assert got_sha == sha
    assert list(labels) == [709, 885, 889]  # the set's file order
    assert labels[885] == Label.from_dict(RECORDS[1])


def test_an_unknown_set_says_where_it_looked(labels_root):
    with pytest.raises(FileNotFoundError, match="no label set"):
        loader.load_flattened("nope")


def test_a_set_edited_in_place_is_refused(labels_root):
    store.write_label_set("tiny", RECORDS, {}, root=labels_root)
    rows_p = labels_root / "tiny" / store.LABEL_SET_ROWS_NAME
    rows_p.write_text(rows_p.read_text().replace("Spirulina", "Chlorella"))

    with pytest.raises(ValueError, match="edited in place"):
        loader.load_flattened("tiny")


def test_a_set_is_parsed_once(labels_root):
    store.write_label_set("tiny", RECORDS, {}, root=labels_root)
    loader.load_flattened("tiny")
    (labels_root / "tiny" / store.LABEL_SET_ROWS_NAME).write_text("")

    assert len(loader.load_flattened("tiny")[1]) == 3


def test_a_returned_set_can_be_edited_without_damaging_the_cache(labels_root):
    store.write_label_set("tiny", RECORDS, {}, root=labels_root)
    loader.load_flattened("tiny")[1].clear()

    assert len(loader.load_flattened("tiny")[1]) == 3


def test_there_is_only_one_loader():
    assert loader.loader() is loader.loader() is loader.DsldLoader()


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



# --- raw: many, ordered -----------------------------------------------------


def test_load_raw_sorts_ids_by_default(tmp_path):
    """An id list is a set, not a ranking: two callers asking for the same ids
    should get the same answer."""
    root = tmp_path / "dump"
    for i in (709, 885, 889):
        _dump_with(root, i, {"id": i})
    assert [d["id"] for d in loader.load_raw([889, 709], root)] == [709, 889]


def test_load_raw_keeps_caller_order_when_asked(tmp_path):
    """For the rarer case where the list really is a ranking."""
    root = tmp_path / "dump"
    for i in (709, 885, 889):
        _dump_with(root, i, {"id": i})
    assert [d["id"] for d in loader.load_raw([889, 709], root, order="given")] == [889, 709]


def test_load_raw_many_can_be_sorted_or_drawn(tmp_path):
    root = tmp_path / "dump"
    for i in (709, 885, 889):
        _dump_with(root, i, {"id": i})
    assert [d["id"] for d in loader.load_raw([889, 709, 885], root, order="sorted")] == [
        709,
        885,
        889,
    ]
    a = [d["id"] for d in loader.load_raw([889, 709, 885], root, order="random", seed=5)]
    b = [d["id"] for d in loader.load_raw([889, 709, 885], root, order="random", seed=5)]
    assert a == b and sorted(a) == [709, 885, 889]


def test_load_raw_many_names_every_missing_id(tmp_path):
    """A hole in the dump is usually a partial download; finding out at the end
    of a batch beats finding out in the output."""
    root = _dump_with(tmp_path / "dump", 709, {"id": 709})
    with pytest.raises(loader.DSLDNotFound, match="1 of 2"):
        loader.load_raw([709, 99999], root)


def test_load_raw_many_can_skip_missing_ids(tmp_path):
    root = _dump_with(tmp_path / "dump", 709, {"id": 709})
    assert [d["id"] for d in loader.load_raw([709, 99999], root, skip_missing=True)] == [709]


def test_dump_ids_lists_the_population_to_sample_from(tmp_path):
    """Without this there is no way to name the dump's contents, so 'a random
    sample of the dump' cannot be expressed."""
    root = tmp_path / "dump"
    for i in (889, 709, 18162):
        _dump_with(root, i, {"id": i})
    assert loader.dump_ids(root) == [709, 889, 18162]


# --- load_raw dispatches on the argument ------------------------------------


@pytest.fixture
def dump(tmp_path):
    root = tmp_path / "dump"
    for i in (709, 885, 889):
        _dump_with(root, i, {"id": i})
    return root


def test_an_id_returns_one_record(dump):
    assert loader.load_raw(709, dump)["id"] == 709


def test_ids_return_a_list_of_records(dump):
    assert [d["id"] for d in loader.load_raw([889, 709], dump)] == [709, 889]


def test_a_string_id_is_one_record_not_three_characters(dump):
    """The trap in dispatching on type: `str` is iterable, so a naive check
    would read "709" as ids '7', '0', '9'. Ids arrive as strings from CLI
    flags and CSV cells, so this is a certainty rather than a corner case."""
    assert loader.load_raw("709", dump)["id"] == 709


def test_any_iterable_of_ids_works(dump):
    """A caller holding a tuple or a generator should not have to build a
    list first."""
    assert [d["id"] for d in loader.load_raw((889, 709), dump)] == [709, 889]
    assert [d["id"] for d in loader.load_raw((i for i in [885, 709]), dump)] == [709, 885]


def test_a_one_element_list_still_returns_a_list(dump):
    """The argument decides the shape, not the count: a caller looping over a
    list of length one must not get a bare dict back."""
    got = loader.load_raw([709], dump)
    assert isinstance(got, list) and len(got) == 1


def test_ordering_one_id_is_refused(dump):
    """It would do nothing; the caller believes they asked for something."""
    with pytest.raises(ValueError, match="more than one id"):
        loader.load_raw(709, dump, order="sorted")


# --- n=: a draw from the dump, without naming ids ---------------------------


def test_n_draws_that_many_records(dump):
    assert len(loader.load_raw(n=2, root=dump)) == 2


def test_n_sorted_takes_the_lowest_ids(dump):
    assert [d["id"] for d in loader.load_raw(n=2, root=dump, order="sorted")] == [709, 885]


def test_n_random_is_reproducible_and_a_real_subset(dump):
    a = [d["id"] for d in loader.load_raw(n=2, root=dump, order="random", seed=9)]
    b = [d["id"] for d in loader.load_raw(n=2, root=dump, order="random", seed=9)]
    assert a == b
    assert len(a) == 2 and set(a) <= {709, 885, 889}


def test_an_unseeded_random_draw_is_refused(dump):
    with pytest.raises(ValueError, match="explicit seed"):
        loader.load_raw(n=2, root=dump, order="random")


def test_n_never_reports_missing_ids(dump):
    """It drew from what the dump holds, so `n` means n records -- not n
    attempts of which some fail."""
    assert len(loader.load_raw(n=3, root=dump)) == 3


def test_n_larger_than_the_dump_returns_what_exists(dump):
    assert len(loader.load_raw(n=99, root=dump)) == 3


def test_n_together_with_ids_is_refused(dump):
    """Two different ways of saying which records; honouring both is undefined."""
    with pytest.raises(ValueError, match="n= draws from the dump"):
        loader.load_raw([709], dump, n=2)


def test_load_raw_with_neither_ids_nor_n_is_refused(dump):
    with pytest.raises(ValueError, match="needs an id"):
        loader.load_raw(root=dump)


