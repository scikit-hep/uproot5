# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("pandas")
pytest.importorskip("awkward_pandas")


def test():
    for arrays in uproot.iterate(
        [skhep_testdata.data_path("uproot-HZZ-uncompressed.root") + ":events"] * 2,
        ["Muon_Px", "Jet_Px", "MET_px"],
        library="pd",
    ):
        pass
