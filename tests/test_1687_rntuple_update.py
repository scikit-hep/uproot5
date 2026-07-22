import os

import awkward as ak
import numpy as np
import pytest
import uproot

try:
    import ROOT

    has_root = True
except ImportError:
    has_root = False


def test_extend_existing_ntuple(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "x": np.array([1, 2, 3, 4, 5], dtype=np.float32),
            "y": np.array([10, 20, 30, 40, 50], dtype=np.int32),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "x": np.array([6, 7, 8], dtype=np.float32),
                "y": np.array([60, 70, 80], dtype=np.int32),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(
            nt["x"].array() == np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        )
        assert ak.all(
            nt["y"].array()
            == np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 8


def test_add_field_ntuple(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3, 4, 5], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"z": np.int32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["z"].array() == np.zeros(5, dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5


def test_add_field_ntuple_duplicate(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="already exists"):
            f["mytuple"].add_fields({"x": np.int32})


def test_extend_ntuple_multiple_times(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend({"x": np.array([4, 5, 6], dtype=np.float32)})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend({"x": np.array([7, 8, 9], dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert ak.all(
            f["mytuple"]["x"].array()
            == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 9


def test_add_multiple_fields_ntuple(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"y": np.int32, "z": np.float64})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert ak.all(
            f["mytuple"]["x"].array() == np.array([1, 2, 3], dtype=np.float32)
        )
        assert ak.all(f["mytuple"]["y"].array() == np.zeros(3, dtype=np.int32))
        assert ak.all(f["mytuple"]["z"].array() == np.zeros(3, dtype=np.float64))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 3


def test_extend_ntuple_wrong_fields(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "x": np.array([1, 2, 3], dtype=np.float32),
            "y": np.array([4, 5, 6], dtype=np.int32),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(Exception):
            f["mytuple"].extend(
                {"x": np.array([7, 8, 9], dtype=np.float32)}
            )  # missing y


def test_ntuple_dtypes(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "x_f32": np.array([1, 2, 3], dtype=np.float32),
            "x_f64": np.array([1, 2, 3], dtype=np.float64),
            "x_i32": np.array([1, 2, 3], dtype=np.int32),
            "x_i64": np.array([1, 2, 3], dtype=np.int64),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields(
            {
                "z_f32": np.float32,
                "z_f64": np.float64,
                "z_i32": np.int32,
                "z_i64": np.int64,
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x_f32"].array() == np.array([1, 2, 3], dtype=np.float32))
        assert ak.all(nt["z_i64"].array() == np.zeros(3, dtype=np.int64))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 3


def test_ntuple_variable_length(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "x": ak.Array([[1, 2], [3, 4, 5], [6]]),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend({"x": ak.Array([[7, 8, 9], [10]])})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert ak.all(
            f["mytuple"]["x"].array()
            == ak.Array([[1, 2], [3, 4, 5], [6], [7, 8, 9], [10]])
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5


def test_ntuple_mixed_types_extend(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {
            "pt": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "jets": ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]),
        }

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "pt": np.array([40.0, 50.0], dtype=np.float32),
                "jets": ak.Array([[7.0, 8.0, 9.0], [10.0]]),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(
            nt["pt"].array() == np.array([10, 20, 30, 40, 50], dtype=np.float32)
        )
        assert ak.all(
            nt["jets"].array() == ak.Array([[1, 2], [3], [4, 5, 6], [7, 8, 9], [10]])
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5


def test_ntuple_add_field_then_extend(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    # add new field (backfilled with zeros)
    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"z": np.int32})

    # now extend with both fields
    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "x": np.array([4, 5, 6], dtype=np.float32),
                "z": np.array([40, 50, 60], dtype=np.int32),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
        assert ak.all(
            nt["z"].array() == np.array([0, 0, 0, 40, 50, 60], dtype=np.int32)
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 6


def test_ntuple_extend_empty(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f.mkrntuple("mytuple", {"x": np.dtype("float32")})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend({"x": np.array([1, 2, 3], dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert nt.num_entries == 3
        assert ak.all(nt["x"].array() == np.array([1, 2, 3], dtype=np.float32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 3


def test_ntuple_multiple_in_file(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["tuple1"] = {"x": np.array([1, 2, 3], dtype=np.float32)}
        f["tuple2"] = {"y": np.array([4, 5, 6], dtype=np.int32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["tuple1"].extend({"x": np.array([4, 5], dtype=np.float32)})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        assert ak.all(
            f["tuple1"]["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32)
        )
        assert ak.all(f["tuple2"]["y"].array() == np.array([4, 5, 6], dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader1 = ROOT.RNTupleReader.Open("tuple1", os.path.join(tmp_path, "test.root"))
        reader2 = ROOT.RNTupleReader.Open("tuple2", os.path.join(tmp_path, "test.root"))
        assert reader1.GetNEntries() == 5
        assert reader2.GetNEntries() == 3


def test_ntuple_multiple_add_fields_then_extend(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"y": np.int32})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"z": np.float64})

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "x": np.array([4, 5, 6], dtype=np.float32),
                "y": np.array([40, 50, 60], dtype=np.int32),
                "z": np.array([400.0, 500.0, 600.0], dtype=np.float64),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
        assert ak.all(
            nt["y"].array() == np.array([0, 0, 0, 40, 50, 60], dtype=np.int32)
        )
        assert ak.all(
            nt["z"].array() == np.array([0, 0, 0, 400, 500, 600], dtype=np.float64)
        )

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 6


def test_ntuple_add_field_and_extend_same_session(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"y": np.int32})
        f["mytuple"].extend(
            {
                "x": np.array([4, 5], dtype=np.float32),
                "y": np.array([40, 50], dtype=np.int32),
            }
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["y"].array() == np.array([0, 0, 0, 40, 50], dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5


def test_ntuple_accept_new_fields(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    # should raise without accept_new_fields
    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="not in this RNTuple"):
            f["mytuple"].extend(
                {
                    "x": np.array([4, 5], dtype=np.float32),
                    "z": np.array([40, 50], dtype=np.int32),
                }
            )

    # with accept_new_fields=True - z backfilled with zeros, then user values
    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].extend(
            {
                "x": np.array([4, 5], dtype=np.float32),
                "z": np.array([40, 50], dtype=np.int32),
            },
            accept_new_fields=True,
        )

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert ak.all(nt["x"].array() == np.array([1, 2, 3, 4, 5], dtype=np.float32))
        assert ak.all(nt["z"].array() == np.array([0, 0, 0, 40, 50], dtype=np.int32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 5


def test_ntuple_add_subfield(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array(
            [
                {"particle": {"pt": 1.0, "eta": 2.0}},
                {"particle": {"pt": 3.0, "eta": 4.0}},
            ]
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"particle.phi": np.float32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert "particle.phi" in nt.keys()
        assert ak.all(nt["particle.phi"].array() == np.zeros(2, dtype=np.float32))
        assert ak.all(nt["particle"].array().pt == np.array([1.0, 3.0]))
        assert ak.all(nt["particle"].array().phi == np.zeros(2, dtype=np.float32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 2


def test_ntuple_add_nested_subfield(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array(
            [
                {"particle": {"track": {"pt": 1.0}}},
                {"particle": {"track": {"pt": 3.0}}},
            ]
        )

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"].add_fields({"particle.track.phi": np.float32})

    with uproot.open(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert "particle.track.phi" in nt.keys()
        assert ak.all(nt["particle.track.phi"].array() == np.zeros(2, dtype=np.float32))

    if has_root and hasattr(ROOT, "RNTupleReader"):
        reader = ROOT.RNTupleReader.Open("mytuple", os.path.join(tmp_path, "test.root"))
        assert reader.GetNEntries() == 2


def test_ntuple_add_subfield_nonexistent_parent(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises(ValueError, match="not found"):
            f["mytuple"].add_fields({"nonexistent.phi": np.float32})


def test_ntuple_add_subfield_typed_parent(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = ak.Array([{"pt": 1.0}, {"pt": 2.0}])

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises((ValueError, TypeError)):
            f["mytuple"].add_fields({"pt.x": np.float32})


def test_ntuple_add_subfield_to_collection(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"jets": ak.Array([[1.0, 2.0], [3.0]])}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        with pytest.raises((ValueError, TypeError)):
            f["mytuple"].add_fields({"jets.x": np.float32})


def test_ntuple_num_entries(tmp_path):
    with uproot.recreate(os.path.join(tmp_path, "test.root")) as f:
        f["mytuple"] = {"x": np.array([1, 2, 3], dtype=np.float32)}

    with uproot.update(os.path.join(tmp_path, "test.root")) as f:
        nt = f["mytuple"]
        assert nt.num_entries == 3
