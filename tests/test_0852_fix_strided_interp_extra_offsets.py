# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import skhep_testdata


def test_xyz_vector_with_headers():
    file = uproot.open(skhep_testdata.data_path("uproot-issue-513.root"))
    trkana = file["TrkAnaNeg/trkana"]
    trkana["demcent/_mom"].array()
