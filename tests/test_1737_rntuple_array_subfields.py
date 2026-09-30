# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")


def _navigate(ntuple, path):
    # Read the top-level field and navigate down to the subfield
    top, *rest = path.split(".")
    array = ntuple[top].array()
    for name in rest:
        array = array[name]
    return array


def _check_all_subfields(ntuple):
    subfields = [key for key in ntuple.keys() if "." in key]
    assert len(subfields) > 0
    for path in subfields:
        expected = _navigate(ntuple, path)
        result = ntuple[path].array()
        assert str(result.type) == str(expected.type), path
        assert ak.array_equal(result, expected, equal_nan=True), path


def test_std_array_of_records():
    filename = skhep_testdata.data_path("test_stl_containers_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        ntuple = f["ntuple"]
        pt = ntuple["array_lv.pt"].array()
        assert str(pt.type) == "5 * 3 * float32"
        assert pt.tolist() == ntuple["array_lv"].array().pt.tolist()
        assert pt.tolist() == [[float(i)] * 3 for i in range(1, 6)]
        _check_all_subfields(ntuple)


def _record(k):
    return {"x": k + 0.5, "y": k}


def _regular(arrays, name, *axes):
    for axis in axes:
        arrays[name] = ak.to_regular(arrays[name], axis=axis)


def test_nested_std_arrays(tmp_path):
    n = 5
    arrays = {
        # std::array<struct>
        "arr": ak.Array([[_record(10 * e + i) for i in range(3)] for e in range(n)]),
        # std::vector<std::array<struct>>
        "vec_arr": ak.Array(
            [
                [[_record(10 * e + i), _record(10 * e + i + 1)] for i in range(e % 3)]
                for e in range(n)
            ]
        ),
        # std::array<std::vector<struct>>
        "arr_vec": ak.Array(
            [
                [[_record(10 * e + j) for j in range((e + i) % 3)] for i in range(2)]
                for e in range(n)
            ]
        ),
        # std::array<std::array<struct>>
        "arr_arr": ak.Array(
            [
                [[_record(10 * e + i), _record(10 * e + i + 1)] for i in range(3)]
                for e in range(n)
            ]
        ),
        # std::array<std::optional<struct>>
        "arr_opt": ak.Array(
            [
                [_record(10 * e + i) if (e + i) % 2 else None for i in range(2)]
                for e in range(n)
            ]
        ),
        # std::array<struct{std::vector<struct>, int}>
        "arr_rec_vec": ak.Array(
            [
                [
                    {"v": [_record(j) for j in range((e + i) % 3)], "w": e + i}
                    for i in range(2)
                ]
                for e in range(n)
            ]
        ),
        # std::array<std::tuple<double, int>>
        "arr_tup": ak.Array(
            [[(e + i + 0.5, e + i) for i in range(2)] for e in range(n)]
        ),
    }
    _regular(arrays, "arr", 1)
    _regular(arrays, "vec_arr", 2)
    _regular(arrays, "arr_vec", 1)
    _regular(arrays, "arr_arr", 1, 2)
    _regular(arrays, "arr_opt", 1)
    _regular(arrays, "arr_rec_vec", 1)
    _regular(arrays, "arr_tup", 1)
    # struct{std::array<struct>, int}
    arrays["rec_arr"] = ak.zip(
        {"a": arrays["arr_arr"][:, 0], "z": ak.Array(list(range(n)))}, depth_limit=1
    )
    # std::vector<struct{std::array<struct>, int}>
    arrays["vec_rec_arr"] = ak.zip(
        {"a": arrays["vec_arr"], "w": ak.num(arrays["vec_arr"], axis=2)},
        depth_limit=2,
    )
    # std::array<struct{std::array<struct>, int}>
    arrays["arr_rec_arr"] = ak.zip(
        {"a": arrays["arr_arr"], "w": ak.num(arrays["arr_arr"], axis=2)},
        depth_limit=2,
    )

    path = os.path.join(tmp_path, "test_1737.root")
    with uproot.recreate(path) as f:
        f.mkrntuple("ntuple", arrays)

    with uproot.open(path) as f:
        ntuple = f["ntuple"]
        assert str(ntuple["arr.x"].array().type) == "5 * 3 * float64"
        assert str(ntuple["arr_vec.y"].array().type) == "5 * 2 * var * int64"
        assert str(ntuple["arr_arr.x"].array().type) == "5 * 3 * 2 * float64"
        assert str(ntuple["rec_arr.a.x"].array().type) == "5 * 2 * float64"
        assert str(ntuple["arr_rec_arr.a.y"].array().type) == "5 * 3 * 2 * int64"
        assert ntuple["arr_tup.1"].array().tolist() == [[e, e + 1] for e in range(n)]
        _check_all_subfields(ntuple)
