# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os
import pickle
import sys

import pytest
import skhep_testdata

import uproot

pytest.importorskip("awkward")


def test_pickle_roundtrip_mmap():
    with uproot.open(skhep_testdata.data_path("uproot-small-dy-withoffsets.root")) as f:
        pkl = pickle.dumps(f["tree"])

    branch = pickle.loads(pkl)["Muon_pt"]
    muonpt1 = branch.array(library="np", entry_start=10, entry_stop=20)
    assert [x.tolist() for x in muonpt1] == [
        [20.60145378112793],
        [50.36957550048828, 41.21387481689453, 3.1869382858276367],
        [51.685970306396484, 35.227813720703125],
        [],
        [],
        [],
        [],
        [23.073759078979492],
        [32.921417236328125, 8.922308921813965, 4.368383407592773],
        [51.9132194519043, 31.930095672607422],
    ]


@pytest.mark.network
def test_pickle_roundtrip_http():
    with uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root") as f:
        pkl = pickle.dumps(f["events"])

    tree = pickle.loads(pkl)
    assert tree.num_entries == 2304
    assert list(tree["M"].array(entry_stop=10)) == [
        82.4626915551,
        83.6262040052,
        83.3084646667,
        82.1493728809,
        90.4691230355,
        89.7576631707,
        89.7739431721,
        90.4855320463,
        91.7737014813,
        91.9488195857,
    ]


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_pickle_roundtrip_xrootd():
    pytest.importorskip("XRootD")
    with uproot.open(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"
    ) as f:
        pkl = pickle.dumps(f["Events"])

    tree = pickle.loads(pkl)
    assert tree.num_entries == 29308627
    assert set(tree["run"].array(entry_stop=10)) == {194778}
