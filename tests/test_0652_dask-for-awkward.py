# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")

pytest.importorskip("pyarrow")  # dask_awkward.lib.testutils needs pyarrow
from dask_awkward.lib.testutils import assert_eq


def test_single_dask_awkward_array():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    ttree = uproot.open(test_path)

    ak_array = ttree.arrays()
    dak_array = uproot.dask(test_path, library="ak")

    assert_eq(dak_array, ak_array)


def test_dask_concatenation():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = skhep_testdata.data_path("uproot-Zmumu-zlib.root") + ":events"
    test_path4 = skhep_testdata.data_path("uproot-Zmumu-lzma.root") + ":events"

    ak_array = uproot.concatenate([test_path1, test_path2, test_path3, test_path4])
    dak_array = uproot.dask(
        [test_path1, test_path2, test_path3, test_path4], library="ak"
    )

    assert_eq(dak_array, ak_array)


def test_multidim_array():
    test_path = (
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root") + ":sample"
    )
    ttree = uproot.open(test_path)

    ak_array = ttree.arrays()
    dak_array = uproot.dask(test_path, library="ak")

    assert len(ak_array) == len(dak_array)
    assert_eq(dak_array, ak_array)


def test_chunking_single_num():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    assert uproot.dask(test_path, step_size=10, library="ak").npartitions == 231


def test_chunking_single_string():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    assert uproot.dask(test_path, step_size="500B", library="ak").npartitions == 330


def test_chunking_multiple_num():
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    filename2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    assert (
        uproot.dask([filename1, filename2], step_size=10, library="ak").npartitions
        == 462
    )


def test_chunking_multiple_string():
    filename1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    filename2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    assert (
        uproot.dask([filename1, filename2], step_size="500B", library="ak").npartitions
        == 922
    )


def test_delay_open():
    test_path1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    test_path2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    test_path3 = skhep_testdata.data_path("uproot-Zmumu-zlib.root") + ":events"
    test_path4 = skhep_testdata.data_path("uproot-Zmumu-lzma.root") + ":events"

    ak_array = uproot.concatenate([test_path1, test_path2, test_path3, test_path4])
    dak_array = uproot.dask(
        [test_path1, test_path2, test_path3, test_path4], open_files=False, library="ak"
    )

    assert_eq(dak_array, ak_array)
