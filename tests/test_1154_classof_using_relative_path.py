# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import uproot
import numpy as np


def test_descend_into_path_classname_of():

    with uproot.recreate("test.root") as f:
        f["Tree"] = {"x": np.array([1, 2, 3, 4, 5])}

    with uproot.open("test.root") as f:
        assert f.classname_of("Tree/x") == "TBranch"
        assert f.title_of("Tree/x") == "x/L"
        assert f.class_of("Tree/x") == uproot.models.TBranch.Model_TBranch
        f.streamer_of("Tree/x")

    # nested directories
    with uproot.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root") as g:
        assert g.classname_of("one/two/tree") == "TTree"
        assert g.classname_of("one/two/tree/Int64") == "TBranch"
