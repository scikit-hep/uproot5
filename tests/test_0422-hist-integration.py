# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import array
import os

import numpy as np
import pytest

import uproot
import uproot.writing

hist = pytest.importorskip("hist")


def test(tmp_path):
    newfile = os.path.join(tmp_path, "newfile.root")

    with uproot.recreate(newfile) as fout:
        fout["h1"] = (
            hist.Hist.new.Reg(10, -5, 5, name="wow", label="wee")
            .Weight()
            .fill([-2, 3, 3, 1, 99], weight=[1, 1, 5, 5, 3])
        )

    with uproot.open(newfile) as fin:
        h1 = fin["h1"]
        assert h1.member("fEntries") == 15
        assert h1.values(flow=True) == pytest.approx(
            [0, 0, 0, 0, 1, 0, 0, 5, 0, 6, 0, 3]
        )

        assert h1.axis().member("fName") == "wow"
        assert h1.axis().member("fTitle") == "wee"
        assert h1.axis().member("fXmin") == -5
        assert h1.axis().member("fXmax") == 5
        assert len(h1.axis().member("fXbins")) == 0
