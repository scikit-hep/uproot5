# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import awkward as ak
import skhep_testdata
import numpy as np


def test_new_support_RNTuple_split_int32_reading():
    with uproot.open(skhep_testdata.data_path("test_ntuple_int_5e4.root")) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert len(df) == 5e4
        assert len(df.one_integers) == 5e4
        assert np.all(df.one_integers == np.arange(5e4 + 1)[::-1][:-1])


def test_new_support_RNTuple_bit_bool_reading():
    with uproot.open(skhep_testdata.data_path("test_ntuple_bit.root")) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert np.all(df.one_bit == np.asarray([1, 0, 0, 1, 0, 0, 1, 0, 0, 1]))


def test_new_support_RNTuple_split_int16_reading():
    with uproot.open(
        skhep_testdata.data_path("test_ntuple_int_multicluster.root")
    ) as f:
        obj = f["ntuple"]
        df = obj.arrays()
        assert len(df.one_integers) == 1e8
        assert df.one_integers[0] == 2
        assert df.one_integers[-1] == 1
        assert np.all(np.unique(df.one_integers[: len(df.one_integers) // 2]) == [2])
        assert np.all(np.unique(df.one_integers[len(df.one_integers) / 2 + 1 :]) == [1])


pytest.importorskip("cramjam")


@pytest.mark.skip(reason="Need to find a similar file in RNTuple RC2 format")
def test_new_support_RNTuple_event_data():
    with uproot.open(
        "https://xrootd-local.unl.edu:1094//store/user/AGC/nanoaod-rntuple/zstd/TT_TuneCUETP8M1_13TeV-powheg-pythia8/cmsopendata2015_ttbar_19980_PU25nsData2015v1_76X_mcRun2_asymptotic_v12_ext3-v1_00000_0000.root"
    ) as f:
        obj = f["Events"]
        df = obj.arrays(["nTau"])
        assert len(df) == 1334428
        assert ak.to_list(df["nTau"][:10]) == [
            0,
            0,
            2,
            0,
            1,
            1,
            1,
            1,
            2,
            0,
        ]
