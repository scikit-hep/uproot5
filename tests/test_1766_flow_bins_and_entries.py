# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os

import numpy as np
import pytest

import uproot
from uproot.writing.identify import (
    to_TAxis,
    to_TH1x,
    to_THashList,
    to_TObjString,
    to_TProfile,
)

hist = pytest.importorskip("hist")


def _write_and_read(tmp_path, obj):
    filepath = os.path.join(tmp_path, "test.root")
    with uproot.recreate(filepath) as f:
        f["h"] = obj
    return uproot.open(filepath)["h"]


def _profile(weighted):
    storage = hist.storage.WeightedMean() if weighted else hist.storage.Mean()
    h = hist.Hist(hist.axis.Regular(2, 0, 2), storage=storage)
    weight = [2.0, 0.5, 1.0] if weighted else None
    h.fill([0.5, 1.5, 1.5], sample=[1.0, 2.0, 4.0], weight=weight)
    return h


def test_unchanged_profile_keeps_fEntries(tmp_path):
    # with weights, ROOT's number of fills is not the sum of weights (3.5)
    h = _profile(True)
    h.metadata = {"fEntries": 3}
    p = _write_and_read(tmp_path, h)
    assert p.member("fEntries") == 3
    assert _write_and_read(tmp_path, p.to_hist()).member("fEntries") == 3


@pytest.mark.parametrize(("weighted", "expected"), [(False, 6), (True, 7)])
def test_added_profiles_recompute_fEntries(tmp_path, weighted, expected):
    p = _write_and_read(tmp_path, _profile(weighted)).to_hist()
    assert _write_and_read(tmp_path, p + p).member("fEntries") == expected


def test_filled_profile_recomputes_fEntries(tmp_path):
    p = _write_and_read(tmp_path, _profile(False)).to_hist()
    p.fill([0.5], sample=[3.0])
    assert _write_and_read(tmp_path, p).member("fEntries") == 4


def test_explicit_fEntries_metadata_is_used(tmp_path):
    p = _profile(True)
    p.metadata = {"fEntries": 7}
    assert _write_and_read(tmp_path, p).member("fEntries") == 7


@pytest.mark.parametrize(
    ("axis", "fills", "expected"),
    [
        (hist.axis.Integer(0, 3, underflow=False), [0, 1, 5, 5], [0, 1, 1, 0, 2]),
        (hist.axis.Integer(0, 3, overflow=False), [-1, 0, 1], [1, 1, 1, 0, 0]),
        (hist.axis.StrCategory(["a", "b"]), ["a", "z", "z"], [0, 1, 0, 2]),
        (hist.axis.IntCategory([1, 2], overflow=False), [1, 1, 2], [0, 2, 1, 0]),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_th1_partial_flow_bins(tmp_path, axis, fills, expected, weighted):
    storage = hist.storage.Weight() if weighted else hist.storage.Double()
    h = hist.Hist(axis, storage=storage)
    h.fill(fills)
    out = _write_and_read(tmp_path, h)
    assert out.values(flow=True).tolist() == expected
    if weighted:
        assert out.variances(flow=True).tolist() == expected


def test_th2_partial_flow_bins(tmp_path):
    h = hist.Hist(
        hist.axis.Integer(0, 2, underflow=False),
        hist.axis.StrCategory(["a", "b"]),
    )
    h.fill([0, 5, 5], ["a", "b", "z"])
    out = _write_and_read(tmp_path, h)
    expected = np.zeros((4, 4))
    expected[1, 1] = 1  # x=0, y="a"
    expected[3, 2] = 1  # x overflow, y="b"
    expected[3, 3] = 1  # x overflow, y overflow
    assert out.values(flow=True).tolist() == expected.tolist()


def _labeled_axis():
    fLabels = to_THashList([to_TObjString(x) for x in "abc"])
    for bin_number, label in enumerate(fLabels, start=1):
        label._bases[0]._members["@fUniqueID"] = bin_number
    return to_TAxis(
        fName="xaxis", fTitle="", fNbins=3, fXmin=0, fXmax=3, fLabels=fLabels
    )


def _labeled_th1(underflow):
    data = np.array([underflow, 1, 0, 0, 0], np.float64)
    return to_TH1x(
        fName="h",
        fTitle="",
        data=data,
        fEntries=data.sum(),
        fTsumw=1.0,
        fTsumw2=1.0,
        fTsumwx=0.5,
        fTsumwx2=0.25,
        fSumw2=data,
        fXaxis=_labeled_axis(),
    )


def _labeled_tprofile(underflow):
    entries = np.array([underflow, 1, 0, 0, 0], np.float64)
    data = 2 * entries
    return to_TProfile(
        fName="p",
        fTitle="",
        data=data,
        fEntries=entries.sum(),
        fTsumw=1.0,
        fTsumw2=1.0,
        fTsumwx=0.5,
        fTsumwx2=0.25,
        fTsumwy=2.0,
        fTsumwy2=4.0,
        fSumw2=2 * data,
        fBinEntries=entries,
        fBinSumw2=np.array([], np.float64),
        fXaxis=_labeled_axis(),
    )


@pytest.mark.parametrize("make", [_labeled_th1, _labeled_tprofile])
def test_labeled_underflow_warns(tmp_path, make):
    h = _write_and_read(tmp_path, make(0.0))
    assert h.to_hist().axes[0].traits.discrete  # no warning when empty

    h = _write_and_read(tmp_path, make(1.0))
    with pytest.warns(UserWarning, match="underflow bin of labeled axis 'xaxis'"):
        h.to_hist()


def test_labels_are_not_shared(tmp_path):
    h = _write_and_read(tmp_path, _labeled_th1(0.0))
    axis = h.member("fXaxis")
    axis.labels().append("d")
    assert axis.labels() == ["a", "b", "c"]
    assert list(axis) == ["a", "b", "c"]
