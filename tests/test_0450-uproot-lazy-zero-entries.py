# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot

pytest.importorskip("awkward")


def test(tmp_path):
    filename = os.path.join(str(tmp_path), "whatever.root")

    with uproot.recreate(filename) as file_for_writing:
        file_for_writing["tree"] = {"branch": []}

    assert uproot.lazy(filename + ":tree").tolist() == []
