# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import pytest
import uproot


def test_expressions_in_pandas(tmp_path):
    pandas = pytest.importorskip("pandas")
    filename = os.path.join(tmp_path, "uproot_test_pandas_expressions.root")
    # create tmp file
    with uproot.recreate(filename) as file:
        file["tree"] = {"b1": [1, 5, 9], "b2": [3, 6, 11]}

    with uproot.open(filename) as file:
        tree = file["tree"]

        # checking different options
        tree.arrays(["log(b1)"], library="pd")  # arbitrary expressions work, like log
        tree.arrays(["where(b1 < b2, b1, b2)"], library="np")  # works with np
        tree.arrays(["where(b1 < b2, b1, b2)"], library="pd")  # should not fail with pd
