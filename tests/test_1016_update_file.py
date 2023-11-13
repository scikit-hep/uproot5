import pytest

import uproot

import numpy as np
import os


def test_file_update(tmp_path):
    file_path = os.path.join(tmp_path, "newfile.root")
    with uproot.recreate(file_path) as f:
        f["tree1"] = {"x": np.array([1, 2, 3])}

    with uproot.update(file_path) as f:
        f["tree2"] = {"y": np.array([4, 5, 6])}

    with uproot.update(file_path) as f:
        f["tree3"] = {"z": np.array([7, 8, 9, 10, 11])}

    # read data and compare
    with uproot.open(file_path) as f:
        assert f["tree1"]["x"].array().tolist() == [1, 2, 3]
        assert f["tree2"]["y"].array().tolist() == [4, 5, 6]
        assert f["tree3"]["z"].array().tolist() == [7, 8, 9, 10, 11]
