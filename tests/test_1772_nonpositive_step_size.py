# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
import skhep_testdata

import uproot


@pytest.fixture
def ttree():
    with uproot.open(skhep_testdata.data_path("uproot-HZZ.root")) as file:
        yield file["events"]


@pytest.fixture
def rntuple(tmp_path):
    filename = tmp_path / "ntuple.root"
    with uproot.recreate(filename) as file:
        file["ntuple"] = {"x": np.arange(10, dtype=np.float64)}
    with uproot.open(filename) as file:
        yield file["ntuple"]


@pytest.mark.parametrize("step_size", [0, -1, -100, np.int64(-1)])
def test_ttree_iterate_rejects_nonpositive_step_size(ttree, step_size):
    with pytest.raises(ValueError, match="step_size"):
        next(ttree.iterate(["NJet"], step_size=step_size))


@pytest.mark.parametrize("step_size", [0, -1])
def test_rntuple_iterate_rejects_nonpositive_step_size(rntuple, step_size):
    with pytest.raises(ValueError, match="step_size"):
        next(rntuple.iterate(step_size=step_size))


@pytest.mark.parametrize("step_size", [0, -1])
def test_uproot_iterate_rejects_nonpositive_step_size(step_size):
    files = {skhep_testdata.data_path("uproot-HZZ.root"): "events"}
    with pytest.raises(ValueError, match="step_size"):
        next(uproot.iterate(files, ["NJet"], step_size=step_size))


def test_positive_step_size_still_reads_every_entry(ttree, rntuple):
    assert sum(len(a) for a in ttree.iterate(["NJet"], step_size=1000)) == 2421
    assert [len(a) for a in rntuple.iterate(step_size=3)] == [3, 3, 3, 1]
