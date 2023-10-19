# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import numpy
import uproot
import skhep_testdata

pytest.importorskip("hist")
ROOT = pytest.importorskip("ROOT")


def test_axes_of_variances_to_hist_2D_weighted():
    hroot2 = ROOT.TH2F("hroot2", "", 2, 0, 1, 3, 0, 1)
    hroot2.Sumw2()

    for _ in range(1000):
        hroot2.Fill(
            5.0 * numpy.random.random(),
            5.0 * numpy.random.random(),
            numpy.random.random(),
        )

    huproot2 = uproot.from_pyroot(hroot2)
    vuproot = huproot2.variances()
    hhist2 = huproot2.to_hist()
    vhist = hhist2.variances()

    # check variances are equal before and after to_hist
    assert (vuproot == vhist).all()


def test_axes_variances_to_hist_3D_weighted():
    hroot3 = ROOT.TH3F("hroot3", "", 3, 0, 1, 2, 0, 1, 5, 0, 1)
    hroot3.Sumw2()

    for _ in range(2000):
        hroot3.Fill(
            5.0 * numpy.random.random(),
            5.0 * numpy.random.random(),
            5.0 * numpy.random.random(),
            numpy.random.random(),
        )

    huproot3 = uproot.from_pyroot(hroot3)
    vuproot = huproot3.variances()
    hhist3 = huproot3.to_hist()
    vhist = hhist3.variances()

    # check variances are equal before and after to_hist
    assert (vuproot == vhist).all()


def test_users_2d_weighted_histogram():
    with uproot.open(skhep_testdata.data_path("uproot-issue-888.root")) as f:
        h = f["hrecoVsgen_ll_cHel_400mttbar"]
        assert (h.variances() == h.to_hist().variances()).all()
