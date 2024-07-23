import pytest
import skhep_testdata

import uproot

pytest.importorskip("pandas")


def test_decompression_threadpool_executor_for_dask():

    class TestThreadPoolExecutor(uproot.source.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None):
            super().__init__(max_workers=max_workers)
            self.submit_count = 0

        def submit(self, task, /, *args, **kwargs):
            self.submit_count += 1
            super().submit(task, *args, **kwargs)

    implicitexecutor = TestThreadPoolExecutor(max_workers=None)

    a = uproot.dask(
        {skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"): "sample"},
        decompression_executor=implicitexecutor,
    )

    a["i4"].compute()

    assert implicitexecutor.max_workers > 0

    assert implicitexecutor.submit_count > 0

    explicitexecutor = TestThreadPoolExecutor(max_workers=1)

    b = uproot.dask(
        {skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"): "sample"},
        decompression_executor=explicitexecutor,
    )

    b["i4"].compute()

    assert explicitexecutor.max_workers == 1

    assert explicitexecutor.submit_count > 0
