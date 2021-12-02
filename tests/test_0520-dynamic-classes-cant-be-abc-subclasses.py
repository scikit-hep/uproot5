# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pickle

import numpy as np
import pytest
import skhep_testdata

import uproot


def test():
    with open("tests/samples/h_dynamic.pkl", "rb") as f:
        assert len(list(pickle.load(f).axis(0))) == 100
