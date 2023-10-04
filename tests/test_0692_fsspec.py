# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest

import uproot
import uproot.source.fsspec


@pytest.mark.network
def test_open_fsspec_http():
    with uproot.open(
        "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root",
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
def test_open_fsspec_github():
    pytest.skip("not working yet")
    with uproot.open(
        "github://CoffeaTeam:coffea@master/tests/samples/nano_dy.root",
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
def test_open_fsspec_local(tmp_path):
    url = "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root"

    # download file to local
    local_path = str(tmp_path / "nano_dy.root")
    import fsspec

    with fsspec.open(url) as f:
        with open(local_path, "wb") as fout:
            fout.write(f.read())

    with uproot.open(
        local_path,
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
def test_open_fsspec_s3():
    pytest.importorskip("s3fs")

    with uproot.open(
        "s3://pivarski-princeton/pythia_ppZee_run17emb.picoDst.root:PicoDst",
        anon=True,
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Event/Event.mEventId"].array(library="np")
        assert len(data) == 8004


@pytest.mark.network
@pytest.mark.xrootd
def test_open_fsspec_xrootd():
    pytest.importorskip("XRootD")
    with uproot.open(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/run"].array(library="np", entry_stop=20)
        assert len(data) == 20
        assert (data == 194778).all()
