# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pytest

import uproot4


@pytest.mark.network
def test_issue176():
    with uproot4.open(
        "https://starterkit.web.cern.ch/starterkit/data/advanced-python-2019/dalitzdata.root"
    ) as f:
        data = f["tree/Y1"].array(library="np")
        assert len(data) == 100000


@pytest.mark.network
def test_issue121():
    with uproot4.open(
        "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root"
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40
