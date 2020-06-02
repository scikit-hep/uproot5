# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

import numpy
import pytest
import skhep_testdata

import uproot4
import uproot4.interpret.numerical
import uproot4.interpret.library
import uproot4.source.futures


def test_any_basket():
    interpretation = uproot4.interpret.numerical.AsDtype(">i4")

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/i4"] as branch:
        assert branch.basket(0).array(interpretation).tolist() == [
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
        ]
        assert branch.basket(1).array(interpretation).tolist() == [
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
        ]
        assert branch.basket(2).array(interpretation).tolist() == [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
        ]
        assert branch.basket(3).array(interpretation).tolist() == [
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ]
        assert branch.basket(4).array(interpretation).tolist() == [
            13,
            14,
        ]


def test_stitching_arrays():
    interpretation = uproot4.interpret.numerical.AsDtype("i8")
    expectation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    basket_arrays = [[0, 1, 2, 3, 4], [5, 6], [], [7, 8, 9], [10], [11, 12, 13, 14]]
    basket_arrays = [numpy.array(x) for x in basket_arrays]
    entry_offsets = numpy.array([0, 5, 7, 7, 10, 11, 15])
    library = uproot4.interpret.library._libraries["np"]

    for start in range(16):
        for stop in range(15, -1, -1):
            actual = interpretation.final_array(
                basket_arrays, start, stop, entry_offsets, library, None
            )
            assert expectation[start:stop] == actual.tolist()


def test_names_entries_to_ranges_or_baskets():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        out = sample._names_entries_to_ranges_or_baskets(["i4"], 0, 30)
        assert all(x[0] == "i4" for x in out)
        assert [x[2] for x in out] == [0, 1, 2, 3, 4]
        assert [x[3] for x in out] == [
            (6992, 7091),
            (16085, 16184),
            (25939, 26038),
            (35042, 35141),
            (40396, 40475),
        ]


def test_ranges_or_baskets_to_arrays():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        branch = sample["i4"]

        ranges_or_baskets = sample._names_entries_to_ranges_or_baskets(["i4"], 0, 30)
        branchid_interpretation = {
            id(branch): uproot4.interpret.numerical.AsDtype(">i4")
        }
        entry_start, entry_stop = (0, 30)
        decompression_executor = uproot4.source.futures.TrivialExecutor()
        interpretation_executor = uproot4.source.futures.TrivialExecutor()
        array_cache = None
        library = uproot4.interpret.library._libraries["np"]

        output = sample._ranges_or_baskets_to_arrays(
            ranges_or_baskets,
            branchid_interpretation,
            entry_start,
            entry_stop,
            decompression_executor,
            interpretation_executor,
            array_cache,
            library,
        )
        assert output["i4"].tolist() == [
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
        ]


def test_branch_array():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/i4"] as branch:
        assert branch.array(
            uproot4.interpret.numerical.AsDtype(">i4"), library="np"
        ).tolist() == [
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
        ]

        assert branch.array(
            uproot4.interpret.numerical.AsDtype(">i4"),
            entry_start=3,
            entry_stop=-5,
            library="np",
        ).tolist() == [
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]

        assert branch.array(
            uproot4.interpret.numerical.AsDtype(">i4"),
            entry_start=3,
            entry_stop=-5,
            interpretation_executor=uproot4.decompression_executor,
            library="np",
        ).tolist() == [
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]

        with pytest.raises(ValueError):
            branch.array(uproot4.interpret.numerical.AsDtype(">i8"), library="np")


def test_cache():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    ) as f:
        assert f.cache_key == "db4be408-93ad-11ea-9027-d201a8c0beef:/"
        assert f["sample"].cache_key == "db4be408-93ad-11ea-9027-d201a8c0beef:/sample"
        assert (
            f["sample/i4"].cache_key
            == "db4be408-93ad-11ea-9027-d201a8c0beef:/sample:i4"
        )
        i4 = f["sample/i4"]
        assert list(f.array_cache) == []
        i4.array(uproot4.interpret.numerical.AsDtype(">i4"), library="np")
        assert list(f.array_cache) == [
            "db4be408-93ad-11ea-9027-d201a8c0beef:/sample:i4:AsDtype(Bi4(),Li4()):0-30:np"
        ]

    with pytest.raises(OSError):
        i4.array(
            uproot4.interpret.numerical.AsDtype(">i4"), entry_start=3, library="np"
        )

    i4.array(uproot4.interpret.numerical.AsDtype(">i4"), library="np")


def test_pandas():
    pandas = pytest.importorskip("pandas")
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/i4"] as branch:
        series = branch.array(
            uproot4.interpret.numerical.AsDtype(">i4"),
            entry_start=3,
            entry_stop=-5,
            interpretation_executor=uproot4.decompression_executor,
            library="pd",
        )
        assert isinstance(series, pandas.Series)
        assert series.values.tolist() == [
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]
