# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

pytest.importorskip("pandas")


def test_decompression_executor_for_dask():

    class TestExecutor(uproot.source.futures.TrivialExecutor):
        def __init__(self):
            self.submit_count = 0

        def submit(self, task, /, *args, **kwargs):
            self.submit_count += 1
            super().submit(task, *args, **kwargs)

    testexecutor = TestExecutor()

    a = uproot.dask(
        {skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root"): "sample"},
        decompression_executor=testexecutor,
    )

    a["i4"].compute()

    assert testexecutor.submit_count > 0
