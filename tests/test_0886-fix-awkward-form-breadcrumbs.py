# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import uproot


def test_fix_awkward_form_breadcrumbs():
    file = uproot.open("uproot-issue-880.root")
    tree = file["Z"]
    assert tree.num_entries == 116
