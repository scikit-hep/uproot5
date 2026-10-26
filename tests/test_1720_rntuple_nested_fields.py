# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import pytest

import uproot

ak = pytest.importorskip("awkward")


def _nested_rntuple(tmp_path):
    arr = ak.Array(
        [
            {
                "jets": [{"pt": float(j), "eta": 0.1 * j} for j in range(k % 3)],
                "rec": {"x": k, "y": 2 * k, "lst": [k] * (k % 2)},
                "z": 3 * k,
            }
            for k in range(6)
        ]
    )
    path = os.path.join(tmp_path, "nested.root")
    with uproot.recreate(path) as f:
        f["nt"] = arr
    return path + ":nt", arr


def test_a_dotted_key_under_a_collection_selects_the_leaf(tmp_path):
    where, arr = _nested_rntuple(tmp_path)
    nt = uproot.open(where)
    assert nt.keys(filter_name="jets.pt") == ["jets.pt"]
    assert nt.keys(filter_name="rec.lst") == ["rec.lst"]
    assert str(nt.arrays(["jets.pt"]).type) == "6 * {jets: var * {pt: float64}}"
    assert ak.array_equal(nt.arrays(["jets.pt"]).jets.pt, arr.jets.pt)
    assert (
        str(nt.to_akform(filter_name=["jets.pt"])[0].type)
        == "{jets: var * {pt: float64}}"
    )


def test_dask_reads_nested_rntuple_fields(tmp_path):
    dak = pytest.importorskip("dask_awkward")
    where, arr = _nested_rntuple(tmp_path)
    ev = uproot.dask(where)
    assert ak.array_equal(ev.rec.x.compute(), arr.rec.x)
    assert ak.array_equal(ev.jets.pt.compute(), arr.jets.pt)
    assert ak.array_equal(ev.rec.lst.compute(), arr.rec.lst)
    assert ak.array_equal(ev.compute(), arr)
    assert set(next(iter(dak.necessary_columns(ev.rec.x).values()))) == {"rec.x"}
    assert ak.array_equal(
        uproot.dask(where, steps_per_file=3).rec.y.compute(), arr.rec.y
    )
    only_x = uproot.dask(where, filter_name="rec.x").compute()
    assert only_x.fields == ["rec"] and only_x.rec.fields == ["x"]
    assert ak.array_equal(only_x.rec.x, arr.rec.x)


def _columns_read(monkeypatch):
    from uproot.models.RNTuple import Model_ROOT_3a3a_RNTuple

    read = set()
    real = Model_ROOT_3a3a_RNTuple.read_cluster_pages

    def logged(self, cluster_idx, col_idx, *args, **kwargs):
        read.add(col_idx)
        return real(self, cluster_idx, col_idx, *args, **kwargs)

    monkeypatch.setattr(Model_ROOT_3a3a_RNTuple, "read_cluster_pages", logged)
    return read


@pytest.mark.parametrize("open_files", [True, False])
def test_dask_reads_only_the_subfields_it_needs(tmp_path, monkeypatch, open_files):
    dak = pytest.importorskip("dask_awkward")
    where, arr = _nested_rntuple(tmp_path)
    ev = uproot.dask(where, open_files=open_files)
    read = _columns_read(monkeypatch)
    assert ak.array_equal(ev.rec.y.compute(), arr.rec.y)
    assert len(read) == 1
    read.clear()
    assert ak.array_equal(ev.rec.compute(), arr.rec)
    assert len(read) == 4  # x, y, and the offsets and content of lst
    # an unprojected read asks for top-level fields, which cover their subfields
    assert ak.array_equal(ev.compute(optimize_graph=False), arr)
    # a collection's offsets come with its first leaf
    assert set(next(iter(dak.necessary_columns(ev.jets.eta).values()))) == {
        "jets.pt",
        "jets.eta",
    }


def test_graphed_reads_only_the_subfields_it_needs(tmp_path, monkeypatch):
    graphed = pytest.importorskip("graphed")
    from graphed.core.execution import SequentialRunner

    where, arr = _nested_rntuple(tmp_path)
    ev = uproot.graphed(where)
    assert ak.array_equal(ak.Array(ev.session.materialize(ev)), arr)
    assert uproot._graphed.necessary_columns(ev.rec.y) == {"nt": frozenset({"rec.y"})}
    plan = graphed.aggregate_plan(
        ev.rec.y,
        reduce=lambda vals: int(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0,
    )
    read = _columns_read(monkeypatch)
    assert SequentialRunner().run(plan).value == ak.sum(arr.rec.y)
    assert len(read) == 1


def test_a_form_mapping_may_name_rntuple_subfields(tmp_path):
    dak = pytest.importorskip("dask_awkward")
    from uproot._dask import TrivialFormMappingInfo

    where, arr = _nested_rntuple(tmp_path)

    def mapping(form):
        expected = ak.forms.form_with_unique_keys(form, ("<root>",))
        return expected, TrivialFormMappingInfo(expected, subfields=True)

    # known_base_form opens no file, so only the mapping knows these are subfields
    ev = uproot.dask(
        where,
        open_files=False,
        known_base_form=uproot.open(where).to_akform()[0],
        form_mapping=mapping,
    )
    assert set(next(iter(dak.necessary_columns(ev.rec.x).values()))) == {"rec.x"}
    assert ak.array_equal(ev.rec.x.compute(), arr.rec.x)
    assert ak.array_equal(ev.compute(), arr)


def test_leaf_paths_are_spelled_like_rntuple_keys():
    from uproot._dask import _rntuple_leaf_paths

    form = ak.forms.from_dict(
        {
            "class": "RecordArray",
            "fields": ["jets", "v", "n"],
            "contents": [
                {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "RecordArray",
                        "fields": ["pt", "tag"],
                        "contents": [
                            {"class": "NumpyArray", "primitive": "float64"},
                            {
                                "class": "UnionArray",
                                "tags": "i8",
                                "index": "i64",
                                "contents": [
                                    {"class": "NumpyArray", "primitive": "int64"},
                                    {
                                        "class": "RecordArray",
                                        "fields": ["a"],
                                        "contents": [
                                            {"class": "NumpyArray", "primitive": "bool"}
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                },
                {
                    "class": "UnmaskedArray",
                    "content": {"class": "NumpyArray", "primitive": "int32"},
                },
                {"class": "NumpyArray", "primitive": "int64"},
            ],
        }
    )
    assert _rntuple_leaf_paths(form.content("jets"), "jets") == [
        "jets.pt",
        "jets.tag",
        "jets.tag.a",
    ]
    assert _rntuple_leaf_paths(form.content("v"), "v") == ["v"]
    assert _rntuple_leaf_paths(form.content("n"), "n") == ["n"]


def test_rntuple_buffers_are_read_with_their_leaf():
    from uproot._dask import TrivialFormMappingInfo

    form = ak.forms.form_with_unique_keys(
        ak.forms.from_dict(
            {
                "class": "RecordArray",
                "fields": ["jets", "u"],
                "contents": [
                    {
                        "class": "ListOffsetArray",
                        "offsets": "i64",
                        "content": {
                            "class": "RecordArray",
                            "fields": ["pt", "eta"],
                            "contents": [
                                {"class": "NumpyArray", "primitive": "float64"},
                                {"class": "NumpyArray", "primitive": "float64"},
                            ],
                        },
                    },
                    {
                        "class": "UnionArray",
                        "tags": "i8",
                        "index": "i64",
                        "contents": [
                            {"class": "NumpyArray", "primitive": "int64"},
                            {
                                "class": "RecordArray",
                                "fields": ["a"],
                                "contents": [
                                    {"class": "NumpyArray", "primitive": "bool"}
                                ],
                            },
                        ],
                    },
                ],
            }
        ),
        ("<root>",),
    )
    by_field = TrivialFormMappingInfo.build_form_key_to_key(form)
    by_leaf = TrivialFormMappingInfo.build_form_key_to_key(form, subfields=True)
    jets = form.content("jets")
    assert by_field[jets.form_key] == by_field[jets.content.content("eta").form_key]
    assert by_leaf[jets.form_key] == "jets.pt"  # the offsets come with the first leaf
    assert by_leaf[jets.content.content("eta").form_key] == "jets.eta"
    union = form.content("u")
    assert {by_leaf[c.form_key] for c in (union, *union.contents)} == {"u"}
    assert set(by_field.values()) == {None, "jets", "u"}
