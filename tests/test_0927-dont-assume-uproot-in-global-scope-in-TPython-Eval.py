# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import random
import string

import pytest

ROOT = pytest.importorskip("ROOT")


def test():
    name = "".join(random.choices(string.ascii_lowercase, k=10))
    h = ROOT.TProfile3D()
    assert __import__("uproot").from_pyroot(h).values().shape == (1, 1, 1)
