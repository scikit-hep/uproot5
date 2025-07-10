# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest
from utils import run_test_in_pyodide


# Taken from test_0637_setup_tests_for_AwkwardForth.py
@run_test_in_pyodide(test_file="issue367b.root")
@pytest.mark.parametrize("is_forth", [False, True])
def test_00(is_forth):
    import uproot

    with uproot.open("issue367b.root") as file:
        branch = file["tree/weights"]
        interp = uproot.interpretation.identify.interpretation_of(branch, {}, False)
        interp._forth = is_forth
        py = branch.array(interp, library="ak", entry_stop=2)
        assert py[0]["0"][0] == "expskin_FluxUnisim"
        # py[-1] == <STLMap {'expskin_FluxUnisim': [0.944759093019904, 1.0890682745548674, ..., 1.1035170311451232, 0.8873957186284592], ...} at 0x7fbc4c1325e0>
        assert py.layout.form == interp.awkward_form(branch.file)
        if is_forth:
            assert interp._complete_forth_code is not None
