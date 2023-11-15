# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest

import uproot

import skhep_testdata


@pytest.mark.network
def test_issue176():
    pytest.importorskip("aiohttp")

    with uproot.open(
        "https://starterkit.web.cern.ch/starterkit/data/advanced-python-2019/dalitzdata.root"
    ) as f:
        data = f["tree/Y1"].array(library="np")
        assert len(data) == 100000


@pytest.mark.network
def test_issue176_again():
    pytest.importorskip("aiohttp")

    with uproot.open(
        "https://starterkit.web.cern.ch/starterkit/data/advanced-python-2019/dalitzdata.root"
    ) as f:
        data = f["tree"].arrays(["Y1", "Y2"], library="np")
        assert len(data["Y1"]) == 100000
        assert len(data["Y2"]) == 100000


def test_issue121():
    with uproot.open(skhep_testdata.data_path("uproot-issue121.root")) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40
