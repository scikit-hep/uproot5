# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations
import numpy as np
import pytest
import skhep_testdata
import uproot

awkward = pytest.importorskip("awkward")


def test_unequal_baskets(tmp_path):
    path = str(tmp_path / "hzz.root")
    with open(skhep_testdata.data_path("uproot-HZZ.root"), "rb") as src:
        with open(path, "wb") as dst:
            dst.write(src.read())

    with uproot.open(path) as f:
        before = f["events"]["NJet"].array(library="np")
        n = f["events"].num_entries

    with uproot.update(path) as f:
        f["events"].add_branches({"new": np.arange(n, dtype=np.float64)})

    with uproot.open(path) as f:
        np.testing.assert_array_equal(
            f["events"]["new"].array(library="np"), np.arange(n)
        )
        np.testing.assert_array_equal(f["events"]["NJet"].array(library="np"), before)


def test_jagged_clear_error(tmp_path):
    path = str(tmp_path / "x.root")
    with uproot.recreate(path) as f:
        f.mktree("t", {"a": np.int64})
        f["t"].extend({"a": np.arange(3)})
    with uproot.update(path) as f:
        with pytest.raises(NotImplementedError, match="jagged"):
            f["t"].add_branches({"j": awkward.Array([[1.0], [], [2.0, 3.0]])})
