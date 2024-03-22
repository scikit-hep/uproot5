import pytest
import skhep_testdata

import uproot
import uproot.source.file
import uproot.source.fsspec

dask = pytest.importorskip("dask")
dask_awkward = pytest.importorskip("dask_awkward")
dask_distributed = pytest.importorskip("dask.distributed")


@pytest.mark.distributed
@pytest.mark.parametrize(
    "handler",
    [None, uproot.source.file.MemmapSource, uproot.source.fsspec.FSSpecSource],
)
def test_issue_1063(handler):
    file_path = skhep_testdata.data_path("uproot-issue121.root")

    with dask_distributed.Client():
        events = uproot.dask({file_path: "Events"}, handler=handler)
        dask.compute(events.Muon_pt)
