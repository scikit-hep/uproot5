# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest

import uproot


@pytest.mark.network
def test():
    pytest.importorskip("aiohttp")
    with uproot.open(
        {"http://scikit-hep.org/uproot3/examples/HZZ.root": "events"}
    ) as t:
        t["MET_px"].array(library="np")
        t["MET_py"].array(library="np")
