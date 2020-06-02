# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
import uproot4.interpret.library
import uproot4.interpret.jagged
import uproot4.interpret.numerical


def test_arrays():
    awkward1 = pytest.importorskip("awkward1")

    interp_i4 = uproot4.interpret.numerical.AsDtype(">i4")
    interp_f4 = uproot4.interpret.numerical.AsDtype(">f4")
    interp_Ai4 = uproot4.interpret.jagged.AsJagged(
        uproot4.interpret.numerical.AsDtype(">i4")
    )
    interp_Af8 = uproot4.interpret.jagged.AsJagged(
        uproot4.interpret.numerical.AsDtype(">f8")
    )

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample"] as sample:
        result = sample.arrays(
            {"i4": interp_i4, "f4": interp_f4, "Ai4": interp_Ai4, "Af8": interp_Af8},
            how="zip",
        )
        assert result.tolist() == [
            {"i4": -15, "f4": -14.899999618530273, "A": []},
            {"i4": -14, "f4": -13.899999618530273, "A": [{"i4": -15, "f8": -15.0}]},
            {
                "i4": -13,
                "f4": -12.899999618530273,
                "A": [{"i4": -15, "f8": -15.0}, {"i4": -13, "f8": -13.9}],
            },
            {
                "i4": -12,
                "f4": -11.899999618530273,
                "A": [
                    {"i4": -15, "f8": -15.0},
                    {"i4": -13, "f8": -13.9},
                    {"i4": -11, "f8": -12.8},
                ],
            },
            {
                "i4": -11,
                "f4": -10.899999618530273,
                "A": [
                    {"i4": -15, "f8": -15.0},
                    {"i4": -13, "f8": -13.9},
                    {"i4": -11, "f8": -12.8},
                    {"i4": -9, "f8": -11.7},
                ],
            },
            {"i4": -10, "f4": -9.899999618530273, "A": []},
            {"i4": -9, "f4": -8.899999618530273, "A": [{"i4": -10, "f8": -10.0}]},
            {
                "i4": -8,
                "f4": -7.900000095367432,
                "A": [{"i4": -10, "f8": -10.0}, {"i4": -8, "f8": -8.9}],
            },
            {
                "i4": -7,
                "f4": -6.900000095367432,
                "A": [
                    {"i4": -10, "f8": -10.0},
                    {"i4": -8, "f8": -8.9},
                    {"i4": -6, "f8": -7.8},
                ],
            },
            {
                "i4": -6,
                "f4": -5.900000095367432,
                "A": [
                    {"i4": -10, "f8": -10.0},
                    {"i4": -8, "f8": -8.9},
                    {"i4": -6, "f8": -7.8},
                    {"i4": -4, "f8": -6.7},
                ],
            },
            {"i4": -5, "f4": -4.900000095367432, "A": []},
            {"i4": -4, "f4": -3.9000000953674316, "A": [{"i4": -5, "f8": -5.0}]},
            {
                "i4": -3,
                "f4": -2.9000000953674316,
                "A": [{"i4": -5, "f8": -5.0}, {"i4": -3, "f8": -3.9}],
            },
            {
                "i4": -2,
                "f4": -1.899999976158142,
                "A": [
                    {"i4": -5, "f8": -5.0},
                    {"i4": -3, "f8": -3.9},
                    {"i4": -1, "f8": -2.8},
                ],
            },
            {
                "i4": -1,
                "f4": -0.8999999761581421,
                "A": [
                    {"i4": -5, "f8": -5.0},
                    {"i4": -3, "f8": -3.9},
                    {"i4": -1, "f8": -2.8},
                    {"i4": 1, "f8": -1.7},
                ],
            },
            {"i4": 0, "f4": 0.10000000149011612, "A": []},
            {"i4": 1, "f4": 1.100000023841858, "A": [{"i4": 0, "f8": 0.0}]},
            {
                "i4": 2,
                "f4": 2.0999999046325684,
                "A": [{"i4": 0, "f8": 0.0}, {"i4": 2, "f8": 1.1}],
            },
            {
                "i4": 3,
                "f4": 3.0999999046325684,
                "A": [{"i4": 0, "f8": 0.0}, {"i4": 2, "f8": 1.1}, {"i4": 4, "f8": 2.2}],
            },
            {
                "i4": 4,
                "f4": 4.099999904632568,
                "A": [
                    {"i4": 0, "f8": 0.0},
                    {"i4": 2, "f8": 1.1},
                    {"i4": 4, "f8": 2.2},
                    {"i4": 6, "f8": 3.3},
                ],
            },
            {"i4": 5, "f4": 5.099999904632568, "A": []},
            {"i4": 6, "f4": 6.099999904632568, "A": [{"i4": 5, "f8": 5.0}]},
            {
                "i4": 7,
                "f4": 7.099999904632568,
                "A": [{"i4": 5, "f8": 5.0}, {"i4": 7, "f8": 6.1}],
            },
            {
                "i4": 8,
                "f4": 8.100000381469727,
                "A": [{"i4": 5, "f8": 5.0}, {"i4": 7, "f8": 6.1}, {"i4": 9, "f8": 7.2}],
            },
            {
                "i4": 9,
                "f4": 9.100000381469727,
                "A": [
                    {"i4": 5, "f8": 5.0},
                    {"i4": 7, "f8": 6.1},
                    {"i4": 9, "f8": 7.2},
                    {"i4": 11, "f8": 8.3},
                ],
            },
            {"i4": 10, "f4": 10.100000381469727, "A": []},
            {"i4": 11, "f4": 11.100000381469727, "A": [{"i4": 10, "f8": 10.0}]},
            {
                "i4": 12,
                "f4": 12.100000381469727,
                "A": [{"i4": 10, "f8": 10.0}, {"i4": 12, "f8": 11.1}],
            },
            {
                "i4": 13,
                "f4": 13.100000381469727,
                "A": [
                    {"i4": 10, "f8": 10.0},
                    {"i4": 12, "f8": 11.1},
                    {"i4": 14, "f8": 12.2},
                ],
            },
            {
                "i4": 14,
                "f4": 14.100000381469727,
                "A": [
                    {"i4": 10, "f8": 10.0},
                    {"i4": 12, "f8": 11.1},
                    {"i4": 14, "f8": 12.2},
                    {"i4": 16, "f8": 13.3},
                ],
            },
        ]


def test_jagged():
    interpretation = uproot4.interpret.jagged.AsJagged(
        uproot4.interpret.numerical.AsDtype(">i2")
    )

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/Ai2"] as branch:
        result = branch.array(interpretation, library="np")
        result = [[int(y) for y in x] for x in result]
        assert result == [
            [],
            [-15],
            [-15, -13],
            [-15, -13, -11],
            [-15, -13, -11, -9],
            [],
            [-10],
            [-10, -8],
            [-10, -8, -6],
            [-10, -8, -6, -4],
            [],
            [-5],
            [-5, -3],
            [-5, -3, -1],
            [-5, -3, -1, 1],
            [],
            [0],
            [0, 2],
            [0, 2, 4],
            [0, 2, 4, 6],
            [],
            [5],
            [5, 7],
            [5, 7, 9],
            [5, 7, 9, 11],
            [],
            [10],
            [10, 12],
            [10, 12, 14],
            [10, 12, 14, 16],
        ]

        result = branch.array(
            interpretation, entry_start=3, entry_stop=-6, library="np"
        )
        result = [[int(y) for y in x] for x in result]
        assert result == [
            [-15, -13, -11],
            [-15, -13, -11, -9],
            [],
            [-10],
            [-10, -8],
            [-10, -8, -6],
            [-10, -8, -6, -4],
            [],
            [-5],
            [-5, -3],
            [-5, -3, -1],
            [-5, -3, -1, 1],
            [],
            [0],
            [0, 2],
            [0, 2, 4],
            [0, 2, 4, 6],
            [],
            [5],
            [5, 7],
            [5, 7, 9],
        ]


def test_jagged_awkward():
    awkward1 = pytest.importorskip("awkward1")

    interpretation = uproot4.interpret.jagged.AsJagged(
        uproot4.interpret.numerical.AsDtype(">i2")
    )

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/Ai2"] as branch:
        result = branch.array(interpretation)
        assert awkward1.to_list(result) == [
            [],
            [-15],
            [-15, -13],
            [-15, -13, -11],
            [-15, -13, -11, -9],
            [],
            [-10],
            [-10, -8],
            [-10, -8, -6],
            [-10, -8, -6, -4],
            [],
            [-5],
            [-5, -3],
            [-5, -3, -1],
            [-5, -3, -1, 1],
            [],
            [0],
            [0, 2],
            [0, 2, 4],
            [0, 2, 4, 6],
            [],
            [5],
            [5, 7],
            [5, 7, 9],
            [5, 7, 9, 11],
            [],
            [10],
            [10, 12],
            [10, 12, 14],
            [10, 12, 14, 16],
        ]


def test_jagged_pandas():
    pandas = pytest.importorskip("pandas")

    interpretation = uproot4.interpret.jagged.AsJagged(
        uproot4.interpret.numerical.AsDtype(">i2")
    )

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    )["sample/Ai2"] as branch:
        result = branch.array(interpretation, library="pd")
        assert result.index.tolist() == [
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (6, 0),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
            (8, 2),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (11, 0),
            (12, 0),
            (12, 1),
            (13, 0),
            (13, 1),
            (13, 2),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (16, 0),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (18, 2),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (21, 0),
            (22, 0),
            (22, 1),
            (23, 0),
            (23, 1),
            (23, 2),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (26, 0),
            (27, 0),
            (27, 1),
            (28, 0),
            (28, 1),
            (28, 2),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
        ]
        assert result.values.tolist() == [
            -15,
            -15,
            -13,
            -15,
            -13,
            -11,
            -15,
            -13,
            -11,
            -9,
            -10,
            -10,
            -8,
            -10,
            -8,
            -6,
            -10,
            -8,
            -6,
            -4,
            -5,
            -5,
            -3,
            -5,
            -3,
            -1,
            -5,
            -3,
            -1,
            1,
            0,
            0,
            2,
            0,
            2,
            4,
            0,
            2,
            4,
            6,
            5,
            5,
            7,
            5,
            7,
            9,
            5,
            7,
            9,
            11,
            10,
            10,
            12,
            10,
            12,
            14,
            10,
            12,
            14,
            16,
        ]


def test_stl_vector():
    interpretation = uproot4.interpret.jagged.AsJagged(
        uproot4.interpret.numerical.AsDtype(">i4"), header_bytes=10
    )

    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/StlVecI32"] as branch:
        result = branch.array(interpretation, library="np")
        result = [[int(y) for y in x] for x in result]
        assert result == [
            [],
            [1],
            [2, 2],
            [3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [],
            [11],
            [12, 12],
            [13, 13, 13],
            [14, 14, 14, 14],
            [15, 15, 15, 15, 15],
            [16, 16, 16, 16, 16, 16],
            [17, 17, 17, 17, 17, 17, 17],
            [18, 18, 18, 18, 18, 18, 18, 18],
            [19, 19, 19, 19, 19, 19, 19, 19, 19],
            [],
            [21],
            [22, 22],
            [23, 23, 23],
            [24, 24, 24, 24],
            [25, 25, 25, 25, 25],
            [26, 26, 26, 26, 26, 26],
            [27, 27, 27, 27, 27, 27, 27],
            [28, 28, 28, 28, 28, 28, 28, 28],
            [29, 29, 29, 29, 29, 29, 29, 29, 29],
            [],
            [31],
            [32, 32],
            [33, 33, 33],
            [34, 34, 34, 34],
            [35, 35, 35, 35, 35],
            [36, 36, 36, 36, 36, 36],
            [37, 37, 37, 37, 37, 37, 37],
            [38, 38, 38, 38, 38, 38, 38, 38],
            [39, 39, 39, 39, 39, 39, 39, 39, 39],
            [],
            [41],
            [42, 42],
            [43, 43, 43],
            [44, 44, 44, 44],
            [45, 45, 45, 45, 45],
            [46, 46, 46, 46, 46, 46],
            [47, 47, 47, 47, 47, 47, 47],
            [48, 48, 48, 48, 48, 48, 48, 48],
            [49, 49, 49, 49, 49, 49, 49, 49, 49],
            [],
            [51],
            [52, 52],
            [53, 53, 53],
            [54, 54, 54, 54],
            [55, 55, 55, 55, 55],
            [56, 56, 56, 56, 56, 56],
            [57, 57, 57, 57, 57, 57, 57],
            [58, 58, 58, 58, 58, 58, 58, 58],
            [59, 59, 59, 59, 59, 59, 59, 59, 59],
            [],
            [61],
            [62, 62],
            [63, 63, 63],
            [64, 64, 64, 64],
            [65, 65, 65, 65, 65],
            [66, 66, 66, 66, 66, 66],
            [67, 67, 67, 67, 67, 67, 67],
            [68, 68, 68, 68, 68, 68, 68, 68],
            [69, 69, 69, 69, 69, 69, 69, 69, 69],
            [],
            [71],
            [72, 72],
            [73, 73, 73],
            [74, 74, 74, 74],
            [75, 75, 75, 75, 75],
            [76, 76, 76, 76, 76, 76],
            [77, 77, 77, 77, 77, 77, 77],
            [78, 78, 78, 78, 78, 78, 78, 78],
            [79, 79, 79, 79, 79, 79, 79, 79, 79],
            [],
            [81],
            [82, 82],
            [83, 83, 83],
            [84, 84, 84, 84],
            [85, 85, 85, 85, 85],
            [86, 86, 86, 86, 86, 86],
            [87, 87, 87, 87, 87, 87, 87],
            [88, 88, 88, 88, 88, 88, 88, 88],
            [89, 89, 89, 89, 89, 89, 89, 89, 89],
            [],
            [91],
            [92, 92],
            [93, 93, 93],
            [94, 94, 94, 94],
            [95, 95, 95, 95, 95],
            [96, 96, 96, 96, 96, 96],
            [97, 97, 97, 97, 97, 97, 97],
            [98, 98, 98, 98, 98, 98, 98, 98],
            [99, 99, 99, 99, 99, 99, 99, 99, 99],
        ]


def test_pandas_merge():
    pandas = pytest.importorskip("pandas")

    group = uproot4.interpret.library.Pandas().group
    name_interp_branch = [("a", None, None), ("b", None, None), ("c", None, None)]

    a = pandas.Series([1, 2, 3, 4, 5])
    b = pandas.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    c = pandas.Series([100, 200, 300, 400, 500])

    df = group({"a": a, "b": b, "c": c}, name_interp_branch, None)
    assert df.index.tolist() == [0, 1, 2, 3, 4]
    assert df.a.tolist() == [1, 2, 3, 4, 5]
    assert df.b.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert df.c.tolist() == [100, 200, 300, 400, 500]

    df = group({"a": a, "b": b, "c": c}, name_interp_branch, "outer")
    assert df.index.tolist() == [0, 1, 2, 3, 4]
    assert df.a.tolist() == [1, 2, 3, 4, 5]
    assert df.b.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert df.c.tolist() == [100, 200, 300, 400, 500]

    a = pandas.Series(
        [1.1, 2.2, 3.3, 4.4, 5.5],
        index=pandas.MultiIndex.from_arrays([[0, 0, 0, 2, 2], [0, 1, 2, 0, 1]]),
    )
    b = pandas.Series([100, 200, 300])
    c = pandas.Series(
        [1, 2, 3, 4, 5, 6],
        index=pandas.MultiIndex.from_arrays([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]),
    )

    dfs = group({"a": a, "b": b, "c": c}, name_interp_branch, None)
    assert isinstance(dfs, tuple) and len(dfs) == 2
    assert dfs[0].columns.tolist() == ["a", "b"]
    assert dfs[0].index.tolist() == [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1)]
    assert dfs[0].a.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert dfs[0].b.tolist() == [100, 100, 100, 300, 300]
    assert dfs[1].columns.tolist() == ["b", "c"]
    assert dfs[1].index.tolist() == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert dfs[1].b.tolist() == [100, 100, 200, 200, 300, 300]
    assert dfs[1].c.tolist() == [1, 2, 3, 4, 5, 6]

    df = group({"a": a, "b": b, "c": c}, name_interp_branch, "outer")
    assert df.columns.tolist() == ["b", "a", "c"]
    assert df.index.tolist() == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert df.b.tolist() == [100, 100, 100, 200, 200, 300, 300]
    assert df.fillna(999).a.tolist() == [1.1, 2.2, 3.3, 999.0, 999.0, 4.4, 5.5]
    assert df.fillna(999).c.tolist() == [1.0, 2.0, 999.0, 3.0, 4.0, 5.0, 6.0]
