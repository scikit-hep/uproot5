# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

from utils import run_test_in_pyodide


# Taken from test_0610_awkward_form.py
@run_test_in_pyodide(packages=["numpy"], test_file="uproot-HZZ-objects.root")
def test_awkward_array_tvector2_array_forth(selenium):
    import numpy as np

    import uproot

    awk_data = None
    with uproot.open("uproot-HZZ-objects.root")["events/MET"] as tree:
        interp = uproot.interpretation.identify.interpretation_of(tree, {}, False)
        interp._forth = True
        awk_data = tree.array(interp, library="ak")
    assert np.isclose(awk_data[0]["fX"], 5.912771224975586)
    assert np.isclose(awk_data[4]["fY"], -1.3100523948669434)
    assert np.isclose(awk_data[1200]["fX"], 1.9457910060882568)
