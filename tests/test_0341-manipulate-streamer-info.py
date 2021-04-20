# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test_volley(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    f1 = ROOT.TFile(filename, "recreate")
    f1.Close()

    assert set(uproot.open(filename).file.streamers) == set([])

    with uproot.writing.update(filename) as f2:
        f2.file._cascading.streamers.write(f2.file.sink)

    assert set(uproot.open(filename).file.streamers) == set([])

    f3 = ROOT.TFile(filename, "update")
    x = ROOT.TObjString("hello")
    x.Write()
    f3.Close()

    assert set(uproot.open(filename).file.streamers) == set(["TObjString"])

    with uproot.writing.update(filename) as f4:
        f4.file._cascading.streamers.write(f4.file.sink)

    assert set(uproot.open(filename).file.streamers) == set(["TObjString"])

    f5 = ROOT.TFile(filename, "update")
    y = ROOT.TH1F("hey", "there", 100, -5, 5)
    y.Write()
    f5.Close()

    assert set(uproot.open(filename).file.streamers) == set(
        [
            "TObject",
            "TNamed",
            "TH1F",
            "TH1",
            "TAttLine",
            "TAttFill",
            "TAttMarker",
            "TAxis",
            "TAttAxis",
            "THashList",
            "TList",
            "TSeqCollection",
            "TCollection",
            "TString",
            "TObjString",
        ]
    )

    with uproot.writing.update(filename) as f6:
        f6.file._cascading.streamers.write(f6.file.sink)

    assert set(uproot.open(filename).file.streamers) == set(
        [
            "TObject",
            "TNamed",
            "TH1F",
            "TH1",
            "TAttLine",
            "TAttFill",
            "TAttMarker",
            "TAxis",
            "TAttAxis",
            "THashList",
            "TList",
            "TSeqCollection",
            "TCollection",
            "TString",
            "TObjString",
        ]
    )
