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


@pytest.mark.parametrize(
    ("fBuffer", "profile"),
    [
        (BUFFER, False),
        (PROFILE_BUFFER, True),
        ([], False),  # no buffer
        ([-3.0, *BUFFER[1:]], False),  # already filled by ROOT
        ([0.0, *PROFILE_BUFFER[1:]], True),  # emptied
    ],
)
def test_warn_on_unfilled_buffer(tmp_path, fBuffer, profile):
    with uproot.recreate(tmp_path / "h.root") as f:
        f["h"] = _buffered(fBuffer, profile)
    file = uproot.open(tmp_path / "h.root")
    if len(fBuffer) > 0 and fBuffer[0] > 0:
        with pytest.warns(
            UserWarning, match=f"has {fBuffer[0]:.0f} entries in its fill"
        ) as record:
            h = file["h"]
        assert record[0].filename == __file__  # the user's code, not Uproot's
    else:
        h = file["h"]  # and no warning, since pytest makes warnings errors
    # uproot doesn't fill the buffered entries
    assert not np.any(h.values(flow=True))
    assert h.member("fEntries") == 0


def test_warn_on_unfilled_buffer_from_ROOT(tmp_path):
    ROOT = pytest.importorskip("ROOT")
    f = ROOT.TFile(str(tmp_path / "h.root"), "RECREATE")
    hists = [
        ROOT.TH1F("h", "", 10, 0, 0),
        ROOT.TProfile("p", "", 10, 0, 0),
        ROOT.TH1D("flushed", "", 10, 0, 0),
    ]
    for x in [1.0, 2.0, 4.0]:
        hists[0].Fill(x, 0.5)
        hists[1].Fill(x, 2 * x)
        hists[2].Fill(x)
    hists[2].GetBinContent(1)  # ROOT fills the buffered entries here
    for h in hists:
        h.Write()
    f.Close()

    file = uproot.open(tmp_path / "h.root")
    for name, classname in [("h", "TH1F"), ("p", "TProfile")]:
        with pytest.warns(UserWarning, match=f"{classname} '{name}' has 3 entries"):
            assert file[name].member("fEntries") == 0
    assert file["flushed"].member("fEntries") == 3

    # also when converted from PyROOT, where the name is a TString
    h = ROOT.TH1F("h_1769_pyroot", "", 10, 0, 0)
    h.SetDirectory(ROOT.nullptr)
    h.Fill(1.0)
    with pytest.warns(UserWarning, match="TH1F 'h_1769_pyroot' has 1 entries"):
        uproot.from_pyroot(h)
