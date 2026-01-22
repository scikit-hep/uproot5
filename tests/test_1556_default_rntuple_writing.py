# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import json
import os
import queue
import sys

import numpy
import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")
pd = pytest.importorskip("pandas")
import numpy as np


def test_write_type_spec(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple(
            "ntuple", {"a": "int32", "b": "float64", "c": np.int64}
        )
        rntuple.extend(
            {
                "a": np.array([1, 2, 3], dtype=np.int32),
                "b": np.array([1.1, 2.2, 3.3], dtype=np.float64),
                "c": np.array([1, 2, 3], dtype=np.int64),
            }
        )

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple["a"].array().tolist() == [1, 2, 3]
        assert rntuple["b"].array().tolist() == [1.1, 2.2, 3.3]
        assert rntuple["c"].array().tolist() == [1, 2, 3]


def test_write_complex_type_spec(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple("ntuple", {"a": "int32", "b": "float64"})
        rntuple.extend(
            {
                "a": np.array([1, 2, 3], dtype=np.int32),
                "b": np.array([1.1, 2.2, 3.3], dtype=np.float64),
            }
        )

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple["a"].array().tolist() == [1, 2, 3]
        assert rntuple["b"].array().tolist() == [1.1, 2.2, 3.3]


def test_write_dict(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple("ntuple", {"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
        rntuple.extend({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple["a"].array().tolist() == [1, 2, 3, 1, 2, 3]
        assert rntuple["b"].array().tolist() == [1.1, 2.2, 3.3, 1.1, 2.2, 3.3]


def test_write_pandas(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    data = {
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Age": [25, 32, 18, 47],
        "City": ["New York", "Los Angeles", "Chicago", "Houston"],
    }
    df = pd.DataFrame(data)

    with uproot.recreate(filepath) as file:
        r = file.mkrntuple("df", df)
        r.extend(df)


def test_extend_dict_mixed_order(tmp_path):
    filepath = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filepath) as file:
        rntuple = file.mkrntuple("ntuple", {"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
        rntuple.extend({"b": [1.1, 2.2, 3.3], "a": [1, 2, 3]})

    with uproot.open(filepath) as file:
        rntuple = file["ntuple"]
        assert rntuple["a"].array().tolist() == [1, 2, 3, 1, 2, 3]
        assert rntuple["b"].array().tolist() == [1.1, 2.2, 3.3, 1.1, 2.2, 3.3]
