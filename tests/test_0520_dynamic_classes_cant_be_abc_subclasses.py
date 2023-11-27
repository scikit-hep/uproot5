# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pickle
import sys
import os

import uproot
import pytest


@pytest.mark.skipif(
    sys.version_info < (3, 7),
    reason="Dynamic types depend on module __getattr__, a Python 3.7+ feature.",
)
def test_pickle(tests_directory):
    with open(os.path.join(tests_directory, "samples/h_dynamic.pkl"), "rb") as f:
        assert len(list(pickle.load(f).axis(0))) == 100
