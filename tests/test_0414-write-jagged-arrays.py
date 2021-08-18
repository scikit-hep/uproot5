# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")
awkward = pytest.importorskip("awkward")


def test(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile, compression=None) as fout:
        fout.mktree(
            "tree",
            {
                "b1": awkward.types.from_datashape("int32"),
                "b2": awkward.types.from_datashape("2 * float64"),
                "b3": awkward.types.from_datashape("2 * 3 * float64"),
                "b4": awkward.Array([1.1, 2.2, 3.3]).type,
            },
        )

    with uproot.open(newfile) as fin:
        assert fin["tree/b1"].typename == "int32_t"
        assert fin["tree/b2"].typename == "double[2]"
        assert fin["tree/b3"].typename == "double[2][3]"
        assert fin["tree/b4"].typename == "double"
