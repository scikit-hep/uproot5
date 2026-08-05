# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for issue #1688: Pandas global indices.

``uproot.concatenate``/``uproot.iterate`` re-index each file's DataFrames so that
entry numbers are global rather than per-file, but ``how=dict`` was not handled.
``RNTuple.arrays`` built its Pandas index from a cluster-relative entry number.
"""

from __future__ import annotations

import numpy as np
import pytest

import uproot

pytest.importorskip("pandas")


def _write_tree(path, start, stop):
    with uproot.recreate(path) as f:
        tree = f.mktree("t", {"x": np.dtype("int64")})
        tree.extend({"x": np.arange(start, stop)})


@pytest.mark.parametrize("num_files", [1, 2])
def test_concatenate_pandas_how_dict(tmp_path, num_files):
    paths = []
    for i in range(num_files):
        path = str(tmp_path / f"file{i}.root")
        _write_tree(path, 5 * i, 5 * (i + 1))
        paths.append(path)

    result = uproot.concatenate(
        {path: "t" for path in paths}, ["x"], library="pd", how=dict
    )

    assert isinstance(result, dict)
    assert list(result) == ["x"]
    assert result["x"].tolist() == list(range(5 * num_files))
    # the index must be global, not restarted at 0 for every file
    assert result["x"].index.tolist() == list(range(5 * num_files))


def test_iterate_pandas_how_dict(tmp_path):
    paths = []
    for i in range(2):
        path = str(tmp_path / f"file{i}.root")
        _write_tree(path, 5 * i, 5 * (i + 1))
        paths.append(path)

    chunks = list(
        uproot.iterate(
            {path: "t" for path in paths}, ["x"], library="pd", how=dict, step_size=5
        )
    )

    assert [type(chunk) for chunk in chunks] == [dict, dict]
    assert chunks[0]["x"].index.tolist() == [0, 1, 2, 3, 4]
    assert chunks[1]["x"].index.tolist() == [5, 6, 7, 8, 9]


@pytest.mark.parametrize("how", [tuple, list])
def test_concatenate_pandas_how_tuple_and_list_still_work(tmp_path, how):
    paths = []
    for i in range(2):
        path = str(tmp_path / f"file{i}.root")
        _write_tree(path, 5 * i, 5 * (i + 1))
        paths.append(path)

    result = uproot.concatenate(
        {path: "t" for path in paths}, ["x"], library="pd", how=how
    )

    assert isinstance(result, how)
    assert result[0].index.tolist() == list(range(10))


def _write_rntuple(path):
    with uproot.recreate(path) as f:
        ntuple = f.mkrntuple("nt", {"x": np.dtype("int64")})
        ntuple.extend({"x": np.arange(0, 4)})
        ntuple.extend({"x": np.arange(4, 8)})


@pytest.mark.parametrize(
    ("entry_start", "entry_stop"),
    [(0, 8), (0, 3), (2, 6), (4, 6), (5, 8), (6, 7)],
)
def test_rntuple_pandas_index_is_global(tmp_path, entry_start, entry_stop):
    path = str(tmp_path / "ntuple.root")
    _write_rntuple(path)

    with uproot.open(path) as f:
        df = f["nt"].arrays(
            library="pd", entry_start=entry_start, entry_stop=entry_stop
        )

    assert df.index.tolist() == list(range(entry_start, entry_stop))
    assert df["x"].tolist() == list(range(entry_start, entry_stop))


def test_rntuple_pandas_index_matches_ttree(tmp_path):
    rntuple_path = str(tmp_path / "ntuple.root")
    ttree_path = str(tmp_path / "tree.root")
    _write_rntuple(rntuple_path)
    with uproot.recreate(ttree_path) as f:
        tree = f.mktree("t", {"x": np.dtype("int64")})
        tree.extend({"x": np.arange(0, 4)})
        tree.extend({"x": np.arange(4, 8)})

    with uproot.open(rntuple_path) as f:
        from_rntuple = f["nt"].arrays(library="pd", entry_start=4, entry_stop=6)
    with uproot.open(ttree_path) as f:
        from_ttree = f["t"].arrays(library="pd", entry_start=4, entry_stop=6)

    assert from_rntuple.index.tolist() == from_ttree.index.tolist() == [4, 5]
