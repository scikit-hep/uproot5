# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot


@pytest.fixture
def tree(tmp_path):
    path = tmp_path / "t.root"
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": np.int64, "y": np.int64})
        f["tree"].extend({"x": np.arange(5), "y": np.arange(5) * 10})
    with uproot.open(path) as f:
        yield f["tree"]


def test_iterate_open_ttree(tree):
    (arrays,) = uproot.iterate(tree, library="np")
    assert arrays["x"].tolist() == [0, 1, 2, 3, 4]
    assert arrays["y"].tolist() == [0, 10, 20, 30, 40]


def test_concatenate_open_ttree(tree):
    arrays = uproot.concatenate(tree, library="np")
    assert arrays["x"].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("read", ["iterate", "concatenate"])
def test_filter_branch_reaches_open_ttree(tree, read):
    keep_y = lambda branch: branch.name == "y"  # noqa: E731
    if read == "iterate":
        (arrays,) = uproot.iterate(tree, filter_branch=keep_y, library="np")
    else:
        arrays = uproot.concatenate(tree, filter_branch=keep_y, library="np")
    assert list(arrays) == ["y"]
