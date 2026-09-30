# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")

NESTED = "test_nested_structs_rntuple_v1-0-0-0.root"
STL = "test_stl_containers_rntuple_v1-0-0-0.root"
VLV = "test_int_vfloat_tlv_vtlv_rntuple_v1-0-0-0.root"

FULL_MY_STRUCT = (
    "{i: int32, sub_struct: {i: int32, sub_sub_struct: {i: int32, v: var * int32}}}"
)


def _ntuple(filename):
    return uproot.open(skhep_testdata.data_path(filename))["ntuple"]


@pytest.mark.parametrize(
    "selection",
    [
        {"filter_name": "my_struct"},
        {"expressions": ["my_struct"]},
        # selecting a field and some of its subfields still gives the whole field
        {"filter_field": lambda f: f.path in ("my_struct", "my_struct.i")},
        {"filter_name": ["my_struct", "my_struct.sub_struct.i"]},
        {"filter_name": "my_struct*"},
    ],
)
def test_selected_field_includes_all_subfields(selection):
    ntuple = _ntuple(NESTED)
    arrays = ntuple.arrays(**selection)
    assert str(arrays.type) == f"10 * {{my_struct: {FULL_MY_STRUCT}}}"
    assert ak.array_equal(arrays.my_struct, ntuple["my_struct"].array())


def test_selected_subfield_includes_its_subfields():
    ntuple = _ntuple(NESTED)
    arrays = ntuple.arrays(filter_name="my_struct.sub_struct")
    assert (
        str(arrays.type)
        == "10 * {my_struct: {sub_struct: {i: int32, sub_sub_struct: {i: int32, v: var * int32}}}}"
    )
    assert ak.array_equal(
        arrays.my_struct.sub_struct, ntuple["my_struct.sub_struct"].array()
    )
    # the same applies when reading from a field
    assert ak.array_equal(ntuple["my_struct"].arrays(filter_name="sub_struct"), arrays)


def test_selected_leaf_keeps_only_the_path_to_it():
    ntuple = _ntuple(NESTED)
    assert (
        str(ntuple.arrays(filter_name="my_struct.sub_struct.i").type)
        == "10 * {my_struct: {sub_struct: {i: int32}}}"
    )
    assert (
        str(
            ntuple.arrays(
                filter_field=lambda f: f.path
                in ("my_struct.i", "my_struct.sub_struct.sub_sub_struct.v")
            ).type
        )
        == "10 * {my_struct: {i: int32, sub_struct: {sub_sub_struct: {v: var * int32}}}}"
    )


def test_keys_are_not_expanded():
    ntuple = _ntuple(NESTED)
    assert ntuple.keys(filter_name="my_struct") == ["my_struct"]
    assert ntuple.keys(filter_name="my_struct.sub_struct") == ["my_struct.sub_struct"]


@pytest.mark.parametrize(
    ("filename", "name", "expected"),
    [
        (STL, "tuple_int32_string", "(int32, string)"),
        (STL, "vector_tuple_int32_string", "var * (int32, string)"),
        (
            STL,
            "array_lv",
            "3 * {pt: float32, eta: float32, phi: float32, mass: float32}",
        ),
        (
            STL,
            "vector_variant_int64_string",
            "var * union[?unknown, ?int64, ?string]",
        ),
        (VLV, "three_LV", "{pt: float32, eta: float32, phi: float32, mass: float32}"),
        (
            VLV,
            "four_v_LVs",
            "var * {pt: float32, eta: float32, phi: float32, mass: float32}",
        ),
    ],
)
def test_selected_containers_include_all_subfields(filename, name, expected):
    ntuple = _ntuple(filename)
    arrays = ntuple.arrays(filter_name=name)
    assert arrays.fields == [name]
    assert str(arrays[name].type.content) == expected
    assert ak.array_equal(arrays[name], ntuple[name].array())


def test_selected_subfield_of_collection():
    ntuple = _ntuple(VLV)
    assert (
        str(ntuple.arrays(filter_name="four_v_LVs.pt").type)
        == "5 * {four_v_LVs: var * {pt: float32}}"
    )
    arrays = ntuple.arrays(filter_name=["four_v_LVs", "four_v_LVs.pt"])
    assert ak.array_equal(arrays.four_v_LVs, ntuple["four_v_LVs"].array())


@pytest.mark.parametrize(
    "selection",
    [
        {"filter_name": "my_struct"},
        {"filter_name": "my_struct.sub_struct"},
        {"filter_name": "my_struct.sub_struct.i"},
        {"filter_field": lambda f: f.path in ("my_struct", "my_struct.i")},
    ],
)
def test_iterate_matches_arrays(selection):
    ntuple = _ntuple(NESTED)
    expected = ntuple.arrays(**selection)
    iterated = ak.concatenate(list(ntuple.iterate(step_size=3, **selection)))
    assert ak.array_equal(iterated, expected)


@pytest.mark.parametrize("open_files", [True, False])
@pytest.mark.parametrize(
    ("filename", "selection"),
    [
        (NESTED, {"filter_name": "my_struct"}),
        (NESTED, {"filter_name": "my_struct.sub_struct"}),
        (NESTED, {"filter_name": "my_struct.sub_struct.i"}),
        (NESTED, {"filter_field": lambda f: f.path in ("my_struct", "my_struct.i")}),
        (STL, {"filter_name": "vector_tuple_int32_string"}),
        (STL, {"filter_name": "lorentz_vector.pt"}),
        (VLV, {"filter_name": "four_v_LVs"}),
        (VLV, {"filter_name": "four_v_LVs.pt"}),
    ],
)
def test_dask_matches_arrays(filename, selection, open_files):
    pytest.importorskip("dask_awkward")
    path = skhep_testdata.data_path(filename)
    expected = uproot.open(path)["ntuple"].arrays(**selection)
    lazy = uproot.dask({path: "ntuple"}, open_files=open_files, **selection)
    assert lazy.fields == expected.fields
    assert ak.array_equal(lazy.compute(), expected)
