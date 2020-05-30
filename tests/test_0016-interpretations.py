# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

import numpy
import pytest
import skhep_testdata

import uproot4


def test_get_key():
    with uproot4.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        assert f["Events"].name == "Events"
        assert f["Events/Info"].name == "Info"
        assert f["Events/Info/evtNum"].name == "evtNum"
        assert f["Events"]["Info/evtNum"].name == "evtNum"
        assert f["Events"]["/Info/evtNum"].name == "evtNum"
        assert f["Events/evtNum"].name == "evtNum"
        assert f["Events"]["evtNum"].name == "evtNum"
        assert f["Events"]["/Info"].name == "Info"
        assert f["Events"]["/Info/"].name == "Info"
        with pytest.raises(KeyError):
            f["Events"]["/evtNum"]


def test_basket_data():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    ) as f:
        assert f["sample/i4"].basket_key(3).fSeekKey == 35042
        assert f["sample/i4"].basket(3).data.view(">i4").tolist() == [
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ]


def test_compressed():
    with uproot4.open(skhep_testdata.data_path("uproot-mc10events.root")) as f:
        basket = f["Events/Muon.q"].basket(0)
        assert basket.data.view(">i4").tolist() == [
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
        ]
        assert basket.byte_offsets.tolist() == [0, 4, 4, 16, 28, 28, 32, 52, 52, 56, 56]


def test_read_all():
    filename = skhep_testdata.data_path("uproot-issue327.root")
    with uproot4.open(filename, minimal_ttree_metadata=False) as f:
        f["DstTree/fTracks.fCharge"]


def test_recovery():
    # uproot-issue327.root DstTree: fTracks.fCharge
    # uproot-issue232.root fTreeV0: V0s.fV0pt MCparticles.nbodies
    # uproot-issue187.root fTreeV0: V0s.fV0pt MCparticles.nbodies
    # uproot-from-geant4.root Details: numgood, TrackedRays: Event phi

    filename = skhep_testdata.data_path("uproot-issue327.root")
    with uproot4.open("file:" + filename + " | DstTree/fTracks.fCharge") as branch:
        print(branch)

    # filename = skhep_testdata.data_path("uproot-issue21.root")
    # with uproot4.open("file:" + filename + " | nllscan", minimal_ttree_metadata=False) as tree:
    #     print(tree)

    # raise Exception
