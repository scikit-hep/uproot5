# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pickle

import skhep_testdata

import uproot


def test():
    with uproot.open(skhep_testdata.data_path("uproot-issue232.root"))["fTreeV0"] as t:
        pickle.loads(pickle.dumps(t["V0Hyper.fNsigmaHe3Pos"]))
