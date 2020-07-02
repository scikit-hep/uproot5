# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy
import pytest
import skhep_testdata

import uproot4


def test_TH2_in_ttree():
    with uproot4.open(skhep_testdata.data_path("uproot-issue-tbranch-of-th2.root"))[
        "g4SimHits/tree"
    ] as tree:
        assert (
            tree["histogram"].array(library="np")[0].member("fXaxis").member("fName")
            == "xaxis"
        )


def test_iofeatures_offsets():
    with uproot4.open(skhep_testdata.data_path("uproot-small-dy-withoffsets.root"))[
        "tree/Muon_pt"
    ] as withoffsets:
        muonpt1 = withoffsets.array(library="np", entry_start=10, entry_stop=20)
        assert [x.tolist() for x in muonpt1] == [
            [51.685970306396484],
            [35.227813720703125, 23.073759078979492, 32.921417236328125],
            [8.922308921813965, 4.368383407592773],
            [],
            [],
            [],
            [],
            [51.9132194519043],
            [31.930095672607422],
            [],
        ]

    with uproot4.open(skhep_testdata.data_path("uproot-small-dy-nooffsets.root"))[
        "tree/Muon_pt"
    ] as nooffsets:
        muonpt2 = nooffsets.array(library="np", entry_start=10, entry_stop=20)
        assert [x.tolist() for x in muonpt2] == [
            [51.685970306396484],
            [35.227813720703125, 23.073759078979492, 32.921417236328125],
            [8.922308921813965, 4.368383407592773],
            [],
            [],
            [],
            [],
            [51.9132194519043],
            [31.930095672607422],
            [],
        ]
