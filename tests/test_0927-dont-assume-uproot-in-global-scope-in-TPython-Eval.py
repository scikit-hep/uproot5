# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import random
import string

import pytest

ROOT = pytest.importorskip("ROOT")


def test():
    name = "".join(random.choices(string.ascii_lowercase, k=10))
    h = ROOT.TH1F(name, "", 100, -5, 5)
    assert len(__import__("uproot").from_pyroot(h).values()) == 100
