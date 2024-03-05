# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import uproot
import skhep_testdata
import numpy as np


def test_descend_into_path_classname_of(tmp_path):
    filename = os.path.join(tmp_path, "test.root")

    with uproot.recreate(filename) as f:
        f["Tree"] = {"x": np.array([1, 2, 3, 4, 5])}

    with uproot.open(filename) as f:
        assert f.classname_of("Tree/x") == "TBranch"
        assert f.title_of("Tree/x").startswith("x/")
        assert f.class_of("Tree/x") == uproot.models.TBranch.Model_TBranch
        f.streamer_of("Tree/x")

    # nested directories
    with uproot.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root") as g:
        assert g.classname_of("one/two/tree") == "TTree"
        assert g.classname_of("one/two/tree/Int64") == "TBranch"

    # check both colon and slash
    with uproot.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    ) as f:
        f.classname_of("tree:evt") == "TBranch"
        f.classname_of("tree/evt") == "TBranch"
