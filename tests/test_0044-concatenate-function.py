# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_concatenate_numpy():
    files = (
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root").replace(
            "6.20.04", "*"
        )
        + ":sample"
    )
    arrays = uproot4.concatenate(files, ["i8", "f8"], library="np")
    assert len(arrays["i8"]) == 420
    assert len(arrays["f8"]) == 420


def test_concatenate_awkward():
    awkward1 = pytest.importorskip("awkward1")
    files = (
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root").replace(
            "6.20.04", "*"
        )
        + ":sample"
    )
    arrays = uproot4.concatenate(files, ["i8", "f8"], library="ak")
    assert isinstance(arrays, awkward1.Array)
    assert set(awkward1.keys(arrays)) == set(["i8", "f8"])
    assert len(arrays) == 420


def test_concatenate_pandas():
    pandas = pytest.importorskip("pandas")
    files = (
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root").replace(
            "6.20.04", "*"
        )
        + ":sample"
    )
    arrays = uproot4.concatenate(files, ["i8", "f8"], library="pd")
    assert isinstance(arrays, pandas.DataFrame)
    assert set(arrays.columns.tolist()) == set(["i8", "f8"])
    assert len(arrays) == 420
