# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pickle

import numpy as np
import pytest
import skhep_testdata

import uproot


@pytest.mark.skipif(
    uproot._util.py26 or uproot._util.py27 or uproot._util.py35 or uproot._util.py36,
    reason="Dynamic types depend on module __getattr__, a Python 3.7+ feature.",
)
def test():
    with open("tests/samples/h_dynamic.pkl", "rb") as f:
        assert len(list(pickle.load(f).axis(0))) == 100
