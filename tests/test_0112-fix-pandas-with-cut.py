# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot

pandas = pytest.importorskip("pandas")


def test():
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root")) as f:
        df = f["events"].arrays(
            ["px1", "pmag"],
            cut="pmag < 30",
            aliases={"pmag": "sqrt(px1**2 + py1**2)"},
            library="pd",
        )
        assert isinstance(df, pandas.DataFrame)  # not an empty tuple
