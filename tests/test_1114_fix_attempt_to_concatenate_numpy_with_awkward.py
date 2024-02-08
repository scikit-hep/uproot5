# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata


def test_partially_fix_issue_951():

    with uproot.open(
        skhep_testdata.data_path("uproot-issue-951.root") + ":CollectionTree"
    ) as tree:
        for key, branch in tree.iteritems(filter_typename="*ElementLink*"):
            try:
                branch.interpretation._forth = True
                branch.array()
            except TypeError as e:
                raise e
            except:
                # ignore for now the two branches which have different issues (basket 0 has the wrong number of bytes) and (uproot.interpretation.identify.UnknownInterpretation: none of the rules matched)
                pass
