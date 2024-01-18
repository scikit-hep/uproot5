# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import uproot
import skhep_testdata


def test_fix_interpretation_for_arrays_of_nonnumerical_objects_issue_880_p2():
    with uproot.open(skhep_testdata.data_path("uproot-issue-880.root")) as file:
        branch = file["Z/Event/Cluster[6]"]
        array = branch.array(library="ak")
        interp = uproot.interpretation.identify.interpretation_of(
            branch, {}, False
        )  # AsObjects(AsArray(False, False, Model_zCluster, (6,)))

        assert len(array) == 116
        assert len(array[0][0]) == 6  # all 6 cluster can now be accessed
