# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata


def test_partially_fix_issue_951():

    with uproot.open(
        skhep_testdata.data_path("uproot-issue-951.root") + ":CollectionTree"
    ) as tree:
        for key, branch in tree.iteritems(filter_typename="*ElementLink*"):
            with pytest.raises(TypeError, match=r".*concat.*"):
                branch.interpretation._forth = True
                branch.array()
