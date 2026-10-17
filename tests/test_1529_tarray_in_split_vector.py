# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import os

import uproot

SAMPLE = os.path.join(os.path.dirname(__file__), "samples", "test_1529.root")


def test_tarray_member_in_split_vector():
    with uproot.open(SAMPLE) as file:
        branch = file["Test/Event/fTracks/fTracks.fPositions"]
        assert branch.array(library="ak").tolist() == [[[1.0, 2.0, 3.0]]]
