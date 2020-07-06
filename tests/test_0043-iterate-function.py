# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_num_entries_for():
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"] as events:
        assert events.num_entries_for("1 kB") == 12
        assert events.num_entries_for("10 kB") == 118
        assert events.num_entries_for("0.1 MB") == 1213
        assert events.num_entries == 2421


def test_num_entries_for_2():
    with uproot4.open(skhep_testdata.data_path("uproot-Zmumu.root"))[
        "events"
    ] as events:
        assert events.num_entries_for("1 kB") == 14
        assert events.num_entries_for("10 kB") == 137
        assert events.num_entries_for("0.1 MB") == 1398
        assert events.num_entries == 2304


def test_iterate_1():
    with uproot4.open(skhep_testdata.data_path("uproot-Zmumu.root"))[
        "events"
    ] as events:
        for i, arrays in enumerate(events.iterate("0.1 MB", library="np")):
            if i == 0:
                assert len(arrays["px1"]) == 1398
            elif i == 1:
                assert len(arrays["px1"]) == 2304 - 1398
            else:
                assert False


def test_iterate_2():
    with uproot4.open(skhep_testdata.data_path("uproot-Zmumu.root"))[
        "events"
    ] as events:
        for i, arrays in enumerate(events.iterate(1000, "px1", library="np")):
            if i == 0:
                assert len(arrays["px1"]) == 1000
                assert arrays["px1"][:5].tolist() == [
                    -41.1952876442,
                    35.1180497674,
                    35.1180497674,
                    34.1444372454,
                    22.7835819537,
                ]
            elif i == 1:
                assert len(arrays["px1"]) == 1000
                assert arrays["px1"][:5].tolist() == [
                    26.043758785,
                    26.043758785,
                    25.9962042016,
                    -44.4626620943,
                    28.2794901505,
                ]
            elif i == 2:
                assert len(arrays["px1"]) == 304
                assert arrays["px1"][:5].tolist() == [
                    -43.3783782352,
                    -43.3783782352,
                    -43.2444221651,
                    -20.2126675303,
                    43.7131175076,
                ]
            else:
                assert False


def test_iterate_pandas_1():
    pandas = pytest.importorskip("pandas")
    with uproot4.open(skhep_testdata.data_path("uproot-Zmumu.root"))[
        "events"
    ] as events:
        for i, arrays in enumerate(events.iterate(1000, "px1", library="pd")):
            if i == 0:
                assert arrays["px1"].index.values[0] == 0
                assert arrays["px1"].index.values[-1] == 999
            elif i == 1:
                assert arrays["px1"].index.values[0] == 1000
                assert arrays["px1"].index.values[-1] == 1999
            elif i == 2:
                assert arrays["px1"].index.values[0] == 2000
                assert arrays["px1"].index.values[-1] == 2303
            else:
                assert False


def test_iterate_pandas_2():
    pandas = pytest.importorskip("pandas")
    with uproot4.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"] as events:
        for i, arrays in enumerate(events.iterate(1000, "Muon_Px", library="pd")):
            if i == 0:
                assert arrays.index.values[0] == (0, 0)
                assert arrays.index.values[-1] == (999, 0)
            elif i == 1:
                assert arrays.index.values[0] == (1000, 0)
                assert arrays.index.values[-1] == (1999, 1)
            elif i == 2:
                assert arrays.index.values[0] == (2000, 0)
                assert arrays.index.values[-1] == (2420, 0)
            else:
                assert False
