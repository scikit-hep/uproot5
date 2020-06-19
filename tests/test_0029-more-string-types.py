# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4


def test_strings1():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree"] as tree:
        result = tree["Beg"].array(library="np")
        assert result.tolist() == ["beg-{0:03d}".format(i) for i in range(100)]

        result = tree["End"].array(library="np")
        assert result.tolist() == ["end-{0:03d}".format(i) for i in range(100)]


def test_map_string_string_in_object():
    with uproot4.open(skhep_testdata.data_path("uproot-issue431.root")) as f:
        print(f["Head"])

    # raise Exception


@pytest.mark.skip(
    reason="FIXME: test works, but the file is not in scikit-hep-testdata yet"
)
def test_map_long_int_in_object():
    with uproot4.open(
        "/home/pivarski/irishep/scikit-hep-testdata/src/skhep_testdata/data/uproot-issue283.root"
    ) as f:
        print(f["config/detector"])

    # raise Exception


# has STL vectors at top-level:
#
# python -c 'import uproot; t = uproot.open("/home/pivarski/irishep/scikit-hep-testdata/src/skhep_testdata/data/uproot-issue38a.root")["ntupler/tree"]; print("\n".join(str((x._fName, getattr(x, "_fStreamerType", None), getattr(x, "_fClassName", None), getattr(x, "_fType", None), x.interpretation)) for x in t.allvalues()))'

# has STL map<int,struct> as described here:
#
# https://github.com/scikit-hep/uproot/issues/468#issuecomment-646325842
#
# python -c 'import uproot; t = uproot.open("/home/pivarski/irishep/scikit-hep-testdata/src/skhep_testdata/data/uproot-issue468.root")["Geant4Data/Geant4Data./Geant4Data.particles"]; print(t.array(uproot.asdebug)[0][:1000])'

# def test_strings1():
#     with uproot4.open(
#         skhep_testdata.data_path("uproot-issue31.root")
#     )["T/name"] as branch:
#         result = branch.array(library="np")
#         assert result.tolist() == ["one", "two", "three", "four", "five"]


@pytest.mark.skip(reason="FIXME: implement strings specified by a TStreamer")
def test_strings2():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/Str"] as branch:
        result = branch.array(library="np")
        assert result.tolist() == ["evt-{0:03d}".format(i) for i in range(100)]


@pytest.mark.skip(reason="FIXME: implement std::string")
def test_strings3():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/StdStr"] as branch:
        result = branch.array(library="np")
        assert result.tolist() == ["std-{0:03d}".format(i) for i in range(100)]
