# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import uproot
import numpy
import hist

ROOT = pytest.importorskip("ROOT")


def test_axes_of_variances_to_hist_2D_weighted():
    hroot2 = ROOT.TH2F("hroot2", "", 2, 0, 1, 3, 0, 1)
    hroot2.Sumw2()

    a = [0.12, 0.32, 0.1, 0.3, 0.4, 0.7, 0.12, 0.12, 0.32, 0.1, 0.3, 0.4, 0.7, 0.12]
    b = [0.22, 0.121, 0.8, 0.3, 0.4, 0.7, 0.23, 0.12, 0.32, 0.1, 0.3, 0.4, 0.7, 0.12]
    c = [23, 19, 17, 11, 7, 5, 3, 23, 19, 17, 11, 7, 5, 3]

    for i in range(14):
        hroot2.Fill(a[i], b[i], c[i])

        # variances with flow:
        # 0.0 0.0 0.0 0.0 0.0
        # 0.0 2329.0 98.0 289.0 0.0
        # 0.0 0.0 0.0 50.0 0.0
        # 0.0 0.0 0.0 0.0 0.0

        huproot2 = uproot.from_pyroot(hroot2)
        vuproot = huproot2.variances()
        hhist2 = huproot2.to_hist()
        vhist = hhist2.variances()
        for i in range(2):
            for j in range(3):
                assert vuproot[i][j] == vhist[i][j]


def test_axes_variances_to_hist_3D_weighted():
    hroot3 = ROOT.TH3F("hroot3", "", 3, 0, 1, 2, 0, 1, 5, 0, 1)
    hroot3.Sumw2()

    a = [0.12, 0.32, 0.1, 0.3, 0.4, 0.7, 0.12, 0.12, 0.32, 0.1, 0.3, 0.4, 0.7, 0.12]
    b = [0.22, 0.121, 0.8, 0.3, 0.4, 0.7, 0.23, 0.12, 0.32, 0.1, 0.3, 0.4, 0.7, 0.12]
    c = [0.22, 0.121, 0.8, 0.3, 0.4, 0.7, 0.23, 0.12, 0.32, 0.1, 0.3, 0.4, 0.7, 0.12]
    d = [23, 19, 17, 11, 7, 5, 3, 23, 19, 17, 11, 7, 5, 3]

    for i in range(14):
        hroot3.Fill(a[i], b[i], c[i], d[i])

    huproot3 = uproot.from_pyroot(hroot3)
    vuproot = huproot3.variances()
    hhist3 = huproot3.to_hist()
    vhist = hhist3.variances()

    # check variances are equal before and after to_hist
    for i in range(3):
        for j in range(2):
            for k in range(5):
                assert vuproot[i][j][k] == vhist[i][j][k]
