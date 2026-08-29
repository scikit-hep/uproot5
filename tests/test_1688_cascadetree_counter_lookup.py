# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for issue #1688: generated TTree counters vs. branch lookup.

When a jagged branch's generated counter name collides with a branch that was
already declared, the colliding datum was deleted from ``_branch_data`` without
reindexing ``_branch_lookup``, so every branch declared after it pointed at the
wrong datum.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

import uproot


def test_branch_lookup_indices_stay_consistent(tmp_path):
    path = str(tmp_path / "file.root")
    with uproot.recreate(path) as f:
        tree = f.mktree(
            "t",
            {
                "nx": np.dtype("int32"),
                "y": np.dtype("float64"),
                "x": ak.Array([[1.1]]).type,
            },
        )
        lookup = tree._cascading._branch_lookup
        data = tree._cascading._branch_data

        # every name must resolve to the datum that carries that name
        assert len(set(lookup.values())) == len(lookup)
        for name, index in lookup.items():
            assert data[index]["fName"] == name

        # the generated counter replaced the scalar 'nx' declared earlier
        assert data[lookup["nx"]]["kind"] == "counter"


def test_write_and_read_back_with_colliding_counter(tmp_path):
    path = str(tmp_path / "file.root")
    x = ak.Array([[1.1], [2.2, 3.3], [4.4, 5.5, 6.6]])
    y = np.array([10.0, 20.0, 30.0])

    with uproot.recreate(path) as f:
        tree = f.mktree(
            "t",
            {"nx": np.dtype("int32"), "y": np.dtype("float64"), "x": x.type},
        )
        tree.extend({"nx": np.array([1, 2, 3], dtype=np.int32), "y": y, "x": x})

    with uproot.open(path) as f:
        result = f["t"].arrays()
        assert result["y"].tolist() == y.tolist()
        assert result["x"].tolist() == x.tolist()
        assert result["nx"].tolist() == [1, 2, 3]


def test_counter_disagreement_still_raises(tmp_path):
    path = str(tmp_path / "file.root")
    x = ak.Array([[1.1], [2.2, 3.3], [4.4, 5.5, 6.6]])

    with uproot.recreate(path) as f:
        tree = f.mktree(
            "t",
            {"nx": np.dtype("int32"), "y": np.dtype("float64"), "x": x.type},
        )
        with pytest.raises(ValueError, match="disagree"):
            tree.extend(
                {
                    "nx": np.array([9, 9, 9], dtype=np.int32),
                    "y": np.array([10.0, 20.0, 30.0]),
                    "x": x,
                }
            )


def test_no_collision_is_unaffected(tmp_path):
    path = str(tmp_path / "file.root")
    x = ak.Array([[1.1], [2.2, 3.3], [4.4, 5.5, 6.6]])
    y = np.array([10.0, 20.0, 30.0])

    with uproot.recreate(path) as f:
        tree = f.mktree("t", {"y": np.dtype("float64"), "x": x.type})
        assert list(tree._cascading._branch_lookup) == ["y", "nx", "x"]
        tree.extend({"y": y, "x": x})

    with uproot.open(path) as f:
        result = f["t"].arrays()
        assert result["y"].tolist() == y.tolist()
        assert result["x"].tolist() == x.tolist()
