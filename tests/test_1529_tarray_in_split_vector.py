# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import skhep_testdata

import uproot


def test_tarray_member_in_split_vector():
    with uproot.open(skhep_testdata.data_path("uproot-issue-1529.root")) as file:
        branch = file["Test/Event/fTracks/fTracks.fPositions"]
        assert branch.array(library="ak").tolist() == [
            [[1.0, 2.0, 3.0], [1.0]],
            [],
            [[]],
            [[1.0, 2.0]],
        ]
