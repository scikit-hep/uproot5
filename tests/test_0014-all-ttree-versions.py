# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import json
import sys
from io import StringIO

import numpy
import pytest
import skhep_testdata

import uproot


def test_streamerless_read():
    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.23.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 16
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 11
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.24.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 16
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 11
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.25.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 17
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.26.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.27.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.28.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.29.02-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 18
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-5.30.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.08.04-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.10.05-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 19
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 12
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.14.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 13
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.16.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 13
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.18.00-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 13
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(
        skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    ) as f:
        assert f["sample"].class_version == 20
        assert isinstance(f["sample"], uproot.TTree)
        assert f["sample"].name == f["sample"].member("fName")
        for x in f["sample"].member("fBranches"):
            assert x.class_version == 13
            assert isinstance(x, uproot.TBranch)
            assert x.name == x.member("fName")
        for x in f["sample"].member("fLeaves"):
            assert x.class_version == 1
        assert f.file._streamers is None

    with uproot.open(skhep_testdata.data_path("uproot-vectorVectorDouble.root")) as f:
        assert f["t"].class_version == 19
        assert f.file._streamers is None


def test_list_streamers():
    with uproot.open(skhep_testdata.data_path("uproot-histograms.root")) as f:
        assert f.file.streamer_dependencies("TNamed") == [
            ("TString", 2),
            ("TObject", 1),
            ("TNamed", 1),
        ]

        output = StringIO()
        f.file.show_streamers("TNamed", stream=output)
        assert (
            output.getvalue()
            == """TString (v2)

TObject (v1)
    fUniqueID: unsigned int (TStreamerBasicType)
    fBits: unsigned int (TStreamerBasicType)

TNamed (v1): TObject (v1)
    fName: TString (TStreamerString)
    fTitle: TString (TStreamerString)
"""
        )
