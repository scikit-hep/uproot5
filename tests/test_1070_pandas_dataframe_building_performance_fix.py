# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

pytest.importorskip("pandas")


def test_pandas_performance_many_branches(tmp_path):
    for array in uproot.iterate(
        skhep_testdata.data_path("uproot-issue-1070.root") + ":tree",
        step_size=100,
        library="pandas",
    ):
        pass
