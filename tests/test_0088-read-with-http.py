# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import pytest

import uproot4


@pytest.mark.network
def test_issue176():
    with uproot4.open(
        "https://starterkit.web.cern.ch/starterkit/data/advanced-python-2019/dalitzdata.root"
    ) as f:
        data = f["tree"].arrays()
        assert len(data.M2AB) == 100000
        assert len(data.M2AC) == 100000
        assert len(data.Y1) == 100000
        assert len(data.Y2) == 100000
        assert len(data.Y3) == 100000


@pytest.mark.network
@pytest.mark.xfail
def test_issue121():
    with uproot4.open(
        "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root"
    ) as f:
        data = f["Runs"].arrays()
        assert len(data) == 1
