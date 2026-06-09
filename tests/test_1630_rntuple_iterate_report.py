# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata

import uproot


def test_rntuple_iterate_report():
    """
    ``RNTuple.iterate(report=True)`` must yield ``(arrays, Report)`` pairs,
    matching the contract documented in its docstring and already provided
    by ``TTree.iterate``.
    """
    path = skhep_testdata.data_path("ntpl001_staff_rntuple_v1-0-0-0.root")
    with uproot.open(path)["Staff"] as rntuple:
        for i, (arrays, report) in enumerate(
            rntuple.iterate("Age", step_size=1000, report=True)
        ):
            if i == 0:
                assert report.tree_entry_start == 0
                assert report.tree_entry_stop == 1000
                assert report.file_path == path
            elif i == 1:
                assert report.tree_entry_start == 1000
                assert report.tree_entry_stop == 2000
                assert report.file_path == path
            elif i == 2:
                assert report.tree_entry_start == 2000
                assert report.tree_entry_stop == 3000
                assert report.file_path == path
            elif i == 3:
                assert report.tree_entry_start == 3000
                assert report.tree_entry_stop == 3354
                assert report.file_path == path
            else:
                assert False
