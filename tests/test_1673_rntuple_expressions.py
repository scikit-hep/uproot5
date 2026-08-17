import pytest
import numpy as np
import uproot
import awkward as ak

formulate = pytest.importorskip("formulate")

def test_rntuple_expressions(tmp_path):
    # tmp_path is a pytest fixture that provides a temporary directory
    filename = tmp_path / "test_rntuple_expressions.root"

    # 1. Create a test RNTuple
    with uproot.recreate(filename) as f:
        f.mkrntuple("ntuple", {"x": np.float64, "y": np.float64, "z": np.float64})
        f["ntuple"].extend({
            "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "y": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            "z": np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        })

    # 2. Test the reading
    with uproot.open(filename) as f:
        rntuple = f["ntuple"]

        # TEST 1: Expressions
        expr_res = rntuple.arrays(["x + y", "x**2"])
        assert expr_res["x + y"].tolist() == [11.0, 22.0, 33.0, 44.0, 55.0]
        assert expr_res["x**2"].tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]

        # TEST 2: Cut (Filtering)
        # Should drop elements where x <= 2.5
        cut_res = rntuple.arrays(["x", "y"], cut="x > 2.5")
        assert cut_res["x"].tolist() == [3.0, 4.0, 5.0]
        assert cut_res["y"].tolist() == [30.0, 40.0, 50.0]

        # TEST 3: Aliases
        alias_res = rntuple.arrays(["my_alias"], aliases={"my_alias": "x * z"})
        assert np.allclose(alias_res["my_alias"].tolist(), [0.1, 0.4, 0.9, 1.6, 2.5])