# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pickle
import os


def test_pickle(test_directory):
    with open(os.path.join(test_directory, "samples/h_dynamic.pkl"), "rb") as f:
        assert len(list(pickle.load(f).axis(0))) == 100
