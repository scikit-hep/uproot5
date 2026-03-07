# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import awkward as ak
import skhep_testdata
import numpy as np


def test_new_support_RNTuple_split_int32_reading():
    with uproot.open(
        skhep_testdata.data_path("test_int_5e4_rntuple_v1-0-0-0.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert len(df) == 5e4
        assert len(df.one_integers) == 5e4
        assert np.all(df.one_integers == np.arange(5e4 + 1)[::-1][:-1])


def test_new_support_RNTuple_split_int32_extended_range():
    values = [0, 1, -1, 2**30, -(2**30), 2**31 - 1, -(2**31)]
    with uproot.open(
        skhep_testdata.data_path("test_splitint32_rntuple_v1-0-1-0.root")
    ) as f:
        result = f["ntuple"]["value"].array().tolist()
    assert result == values


def test_new_support_RNTuple_bit_bool_reading():
    with uproot.open(skhep_testdata.data_path("test_bit_rntuple_v1-0-0-0.root")) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert np.all(df.one_bit == np.asarray([1, 0, 0, 1, 0, 0, 1, 0, 0, 1]))


def test_new_support_RNTuple_split_int16_reading():
    with uproot.open(
        skhep_testdata.data_path("test_int_multicluster_rntuple_v1-0-0-0.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert len(df.one_integers) == 1e8
        assert df.one_integers[0] == 2
        assert df.one_integers[-1] == 1
        assert np.all(np.unique(df.one_integers[: len(df.one_integers) // 2]) == [2])
        assert np.all(np.unique(df.one_integers[len(df.one_integers) / 2 + 1 :]) == [1])


def test_new_support_RNTuple_split_int16_extended_range():
    values = [0, 1, -1, 2**14, -(2**14), 2**15 - 1, -(2**15)]
    with uproot.open(
        skhep_testdata.data_path("test_splitint16_rntuple_v1-0-1-0.root")
    ) as f:
        result = f["ntuple"]["value"].array().tolist()
    assert result == values


def test_new_support_RNTuple_split_int64_extended_range():
    values = [0, 1, -1, 2**62, -(2**62), 2**63 - 1, -(2**63)]
    with uproot.open(
        skhep_testdata.data_path("test_splitint64_rntuple_v1-0-1-0.root")
    ) as f:
        result = f["ntuple"]["value"].array().tolist()
    assert result == values
