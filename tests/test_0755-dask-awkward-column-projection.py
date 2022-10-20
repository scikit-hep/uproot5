# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")

from dask_awkward.lib.testutils import assert_eq


def test_column_projection_sanity_check():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ttree = uproot.open(test_path)

    ak_array = ttree.arrays()
    dak_array = uproot.dask(test_path, library="ak")

    assert_eq(
        dak_array[["px1", "px2", "py1", "py2"]], ak_array[["px1", "px2", "py1", "py2"]]
    )
