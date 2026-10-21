# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import numpy as np
import pytest

import uproot
from uproot.writing.identify import to_TAxis, to_TH1x, to_TProfile

# (w, x) entries, or (w, x, y) for a profile, after the number of entries
BUFFER = [3.0, 1.0, 2.5, 2.0, 0.5, 0.5, 1.5, 0.0, 0.0]
PROFILE_BUFFER = [2.0, 1.0, 2.5, 4.0, 2.0, 0.5, -1.0, 0.0, 0.0, 0.0]


def _buffered(fBuffer, profile=False):
    # an automatically binned histogram (xmin == xmax) with entries in its buffer,
    # as ROOT writes it before they're filled
    xaxis = to_TAxis("xaxis", "", 4, 0.0, 0.0)
    if profile:
        zeros = np.zeros(6)
        return to_TProfile(
            "h", "", zeros, *[0.0] * 7, zeros, zeros, np.zeros(0), xaxis,
            fBufferSize=len(fBuffer), fBuffer=fBuffer,
        )  # fmt: skip
    return to_TH1x(
        "h", "", np.zeros(6), *[0.0] * 5, None, xaxis,
        fBufferSize=len(fBuffer), fBuffer=fBuffer,
    )  # fmt: skip


@pytest.mark.parametrize("fBuffer", [[], [-2.0, 1.0, 0.5, 2.0, 1.5]])
def test_write_fBuffer(tmp_path, fBuffer):
    with uproot.recreate(tmp_path / "h.root") as f:
        f["h"] = _buffered(fBuffer)
    h = uproot.open(tmp_path / "h.root")["h"]
    assert h.member("fBufferSize") == len(fBuffer)
    assert h.member("fBuffer").tolist() == fBuffer

    with pytest.raises(ValueError):
        to_TH1x("h", "", np.zeros(3), *[0.0] * 5, None, to_TAxis("x", "", 1, 0, 1),
                fBuffer=np.ones(3))  # fmt: skip


@pytest.mark.parametrize("profile", [False, True])
def test_write_fBuffer_read_by_ROOT(tmp_path, profile):
    # ROOT reads the buffered entries and bins them when the histogram is used
    ROOT = pytest.importorskip("ROOT")
    fBuffer = PROFILE_BUFFER if profile else BUFFER
    with uproot.recreate(tmp_path / "h.root") as f:
        f["h"] = _buffered(fBuffer, profile)
    f = ROOT.TFile(str(tmp_path / "h.root"))
    h = f.Get("h")
    assert [h.GetBuffer()[i] for i in range(h.GetBufferSize())] == fBuffer
    assert h.GetEntries() == fBuffer[0]
    get = h.GetBinEntries if profile else h.GetBinContent
    weights = fBuffer[
        1 : 1 + int(fBuffer[0]) * (3 if profile else 2) : 3 if profile else 2
    ]
    assert sum(get(i) for i in range(h.GetNcells())) == sum(weights)
    f.Close()
