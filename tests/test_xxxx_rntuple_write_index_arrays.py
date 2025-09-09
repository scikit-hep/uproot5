# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import pytest

import uproot

ak = pytest.importorskip("awkward")

record_array = ak.Array(
    [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}, {"x": 7, "y": 8}]
).layout
record_index_array = ak.contents.IndexedArray(
    ak.index.Index([1, 0, 2, 2]),
    record_array,
)
record_index_option_array = ak.contents.IndexedOptionArray(
    ak.index.Index([1, -1, 2, 2]),
    record_array,
)
record_list_array = ak.contents.ListArray(
    ak.index.Index([1, 0, 2, 2]),
    ak.index.Index([4, 4, 4, 4]),
    record_array,
)

data = ak.Array(
    {
        "record_index_array": record_index_array,
        "record_index_option_array": record_index_option_array,
        "record_list_array": record_list_array,
    }
)


def test_writing_and_reading(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        obj = file.mkrntuple("ntuple", data.layout.form)
        obj.extend(data)
        obj.extend(data)

    obj = uproot.open(filepath)["ntuple"]
    arrays = obj.arrays()

    for f in data.fields:
        if f == "record_index_option_array":
            assert [
                t[0].tolist() if len(t) > 0 else None for t in arrays[f][:4]
            ] == data[f].tolist()
            assert [
                t[0].tolist() if len(t) > 0 else None for t in arrays[f][4:]
            ] == data[f].tolist()
        else:
            assert arrays[f][:4].tolist() == data[f].tolist()
            assert arrays[f][4:].tolist() == data[f].tolist()
