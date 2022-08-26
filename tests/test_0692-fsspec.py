# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest

import uproot
import uproot.source.fsspec


@pytest.mark.network
def test_open_fsspec_http():
    with uproot.open(
        "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root",
        http_handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
@pytest.mark.xrootd
def test_open_fsspec_xrootd():
    pytest.importorskip("XRootD")
    with uproot.open(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        xrootd_handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/run"].array(library="np", entry_stop=20)
        assert len(data) == 20
        assert (data == 194778).all()
