# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import pytest
import numpy as np
import uproot

ROOT = pytest.importorskip("ROOT")


def test_xyz_vector_with_headers():
    file = uproot.open("./TAtest2.root")
    trkana = file["TrkAnaNeg/trkana"]
    trkana["demcent/_mom"].array()
