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


def test_pandas_merge():
    pandas = pytest.importorskip("pandas")
    group = uproot4.interpret.library.Pandas().group

    a = pandas.Series([1, 2, 3, 4, 5])
    b = pandas.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    c = pandas.Series([100, 200, 300, 400, 500])

    df = group({"a": a, "b": b, "c": c}, ["a", "b", "c"], None)
    assert df.index.tolist() == [0, 1, 2, 3, 4]
    assert df.a.tolist() == [1, 2, 3, 4, 5]
    assert df.b.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert df.c.tolist() == [100, 200, 300, 400, 500]

    df = group({"a": a, "b": b, "c": c}, ["a", "b", "c"], "outer")
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

    dfs = group({"a": a, "b": b, "c": c}, ["a", "b", "c"], None)
    assert isinstance(dfs, tuple) and len(dfs) == 2
    assert dfs[0].columns.tolist() == ["a", "b"]
    assert dfs[0].index.tolist() == [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1)]
    assert dfs[0].a.tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert dfs[0].b.tolist() == [100, 100, 100, 300, 300]
    assert dfs[1].columns.tolist() == ["b", "c"]
    assert dfs[1].index.tolist() == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert dfs[1].b.tolist() == [100, 100, 200, 200, 300, 300]
    assert dfs[1].c.tolist() == [1, 2, 3, 4, 5, 6]

    df = group({"a": a, "b": b, "c": c}, ["a", "b", "c"], "outer")
    assert df.columns.tolist() == ["b", "a", "c"]
    assert df.index.tolist() == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert df.b.tolist() == [100, 100, 100, 200, 200, 300, 300]
    assert df.fillna(999).a.tolist() == [1.1, 2.2, 3.3, 999.0, 999.0, 4.4, 5.5]
    assert df.fillna(999).c.tolist() == [1.0, 2.0, 999.0, 3.0, 4.0, 5.0, 6.0]
