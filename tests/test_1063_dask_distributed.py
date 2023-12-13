import pytest
import skhep_testdata

import uproot

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")
dask_distributed = pytest.importorskip("dask.distributed")


def test_issue_1063():
    file_path = skhep_testdata.data_path("uproot-issue121.root")
    with dask_distributed.Client() as _:
        events = uproot.dask({file_path: "Events"})
        print(dask.compute(events.Muon_pt))
