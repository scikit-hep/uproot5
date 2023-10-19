# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata
import numpy as np


def test_new_support_RNTuple_split_int32_reading():
    with uproot.open(
        skhep_testdata.data_path("uproot_ntuple_int_5e4_629_01.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert len(df) == 5e4
        assert len(df.one_integers) == 5e4
        assert np.all(df.one_integers == np.arange(5e4 + 1)[::-1][:-1])


def test_new_support_RNTuple_bit_bool_reading():
    with uproot.open(skhep_testdata.data_path("uproot_ntuple_bit_629_01.root")) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert np.all(df.one_bit == np.asarray([1, 0, 0, 1, 0, 0, 1, 0, 0, 1]))


def test_new_support_RNTuple_split_int16_reading():
    with uproot.open(
        skhep_testdata.data_path("uproot_ntuple_int_multicluster_629_01.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert len(df.one_integers) == 1e8
        assert df.one_integers[0] == 2
        assert df.one_integers[-1] == 1
        assert np.all(np.unique(df.one_integers[: len(df.one_integers) // 2]) == [2])
        assert np.all(np.unique(df.one_integers[len(df.one_integers) / 2 + 1 :]) == [1])
