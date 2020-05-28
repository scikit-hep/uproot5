# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4


def test_streamerless_read():
    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.23.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 16
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 11
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.24.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 16
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 11
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.25.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 17
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.26.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.27.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.28.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.29.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-5.30.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.10.05-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.14.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 13
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 13
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.18.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 13
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 13
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot4.open(skhep_testdata.data_path("uproot-vectorVectorDouble.root")) as f:
        assert f["t"].class_version == 19
        assert f.file._streamers is None

def test_list_streamers():
    with uproot4.open(skhep_testdata.data_path("uproot-histograms.root")) as f:
        pass
