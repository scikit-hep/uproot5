# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import numpy
import pytest
import skhep_testdata

import uproot


def test_axis():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        assert (
            f["hpx"].axes[0]
            == f["hpx"].axis(0)
            == f["hpx"].axis(-1)
            == f["hpx"].axis("x")
        )
        axis = f["hpx"].axis()
        assert len(axis) == 100
        assert axis[0] == (-4.0, -3.92)
        assert axis[1] == (-3.92, -3.84)
        assert [tuple(x) for x in axis][:3] == [
            (-4.0, -3.92),
            (-3.92, -3.84),
            (-3.84, -3.76),
        ]
        assert axis == axis
        assert f["hpxpy"].axis(0) == f["hpxpy"].axis(1)
        assert axis.traits.circular is False
        assert axis.traits.discrete is False
        assert axis.low == -4
        assert axis.high == 4
        assert axis.width == 0.08
        assert axis.labels() is None
        assert axis.edges()[:5].tolist() == [-4.0, -3.92, -3.84, -3.76, -3.68]
        assert axis.edges(True)[:4].tolist() == [-numpy.inf, -4.0, -3.92, -3.84]
        assert axis.intervals()[:2].tolist() == [[-4.0, -3.92], [-3.92, -3.84]]
        assert axis.intervals(True)[:2].tolist() == [[-numpy.inf, -4.0], [-4.0, -3.92]]
        assert axis.centers()[:4].tolist() == [-3.96, -3.88, -3.8, -3.7199999999999998]
        assert axis.centers()[:3].tolist() == [-3.96, -3.88, -3.8]
        assert axis.centers(True)[:3].tolist() == [-numpy.inf, -3.96, -3.88]
        assert axis.widths()[:3].tolist() == [0.08, 0.08, 0.08]
        assert axis.widths(True)[:2].tolist() == [numpy.inf, 0.08000000000000007]
        assert (
            len(axis)
            == len(axis.edges()) - 1
            == len(axis.intervals())
            == len(axis.centers())
            == len(axis.widths())
        )
        assert (
            len(axis) + 2
            == len(axis.edges(flow=True)) - 1
            == len(axis.intervals(flow=True))
            == len(axis.centers(flow=True))
            == len(axis.widths(flow=True))
        )

    with uproot.open(skhep_testdata.data_path("uproot-issue33.root")) as f:
        assert f["cutflow"].axes[0] == f["cutflow"].axis(0) == f["cutflow"].axis("x")
        axis = f["cutflow"].axis()
        assert len(axis) == 7
        assert axis[0] == "Dijet"
        assert axis[1] == "MET"
        assert list(axis)[:3] == ["Dijet", "MET", "MuonVeto"]
        assert axis == axis
        assert axis.traits.circular is False
        assert axis.traits.discrete is True
        assert axis.low == 0.0
        assert axis.high == 7.0
        assert axis.width == 1.0
        assert axis.labels()[:3] == ["Dijet", "MET", "MuonVeto"]
        assert list(axis) == axis.labels()
        assert axis.edges()[:5].tolist() == [0, 1, 2, 3, 4]
        assert axis.edges(True)[:4].tolist() == [-numpy.inf, 0, 1, 2]
        assert axis.intervals()[:2].tolist() == [[0, 1], [1, 2]]
        assert axis.intervals(True)[:2].tolist() == [[-numpy.inf, 0], [0, 1]]
        assert axis.centers()[:4].tolist() == [0.5, 1.5, 2.5, 3.5]
        assert axis.centers(True)[:3].tolist() == [-numpy.inf, 0.5, 1.5]
        assert axis.widths()[:3].tolist() == [1, 1, 1]
        assert axis.widths(True)[:3].tolist() == [numpy.inf, 1, 1]
        assert (
            len(axis.edges()) - 1
            == len(axis.intervals())
            == len(axis.centers())
            == len(axis.widths())
        )
        assert (
            len(axis.edges(flow=True)) - 1
            == len(axis.intervals(flow=True))
            == len(axis.centers(flow=True))
            == len(axis.widths(flow=True))
        )


def test_bins():
    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        for i in range(2):
            hpx = f["hpx"]
            hpxpy = f["hpxpy"]
            hprof = f["hprof"]
            assert len(hpx.axis().centers()) == len(hpx.values())
            assert len(hpx.axis().centers(flow=True)) == len(hpx.values(flow=True))
            assert len(hprof.axis().centers()) == len(hprof.values())
            assert len(hprof.axis().centers(flow=True)) == len(hprof.values(flow=True))
            assert (
                len(hpxpy.axis(0).centers()),
                len(hpxpy.axis(1).centers()),
            ) == hpxpy.values().shape
            assert (
                len(hpxpy.axis(0).centers(flow=True)),
                len(hpxpy.axis(1).centers(flow=True)),
            ) == hpxpy.values(flow=True).shape
            assert numpy.all(hpxpy.values() == hpxpy.variances())
            assert numpy.all(hpxpy.values(flow=True) == hpxpy.variances(flow=True))


def test_boost():
    boost_histogram = pytest.importorskip("boost_histogram")

    with uproot.open(skhep_testdata.data_path("uproot-hepdata-example.root")) as f:
        hpx = f["hpx"]
        hpxpy = f["hpxpy"]
        hprof = f["hprof"]

        assert hpx.to_boost() == boost_histogram.Histogram(hpx)

        assert hpx.to_boost().name == "hpx"
        assert hpx.to_boost().axes[0].name == "xaxis"

        assert hpxpy.to_boost().name == "hpxpy"
        assert hpxpy.to_boost().axes[0].name == "xaxis"
        assert hpxpy.to_boost().axes[1].name == "yaxis"

        assert hprof.to_boost().name == "hprof"
        assert hprof.to_boost().axes[0].name == "xaxis"


@pytest.mark.skip(
    reason="Something's wrong with uproot-issue33.root and boost-histogram"
)
def test_boost_2():
    boost_histogram = pytest.importorskip("boost_histogram")

    with uproot.open(skhep_testdata.data_path("uproot-issue33.root")) as f:
        f["cutflow"].to_boost()
        # assert f["cutflow"].to_boost().name == "cutflow"
        # assert f["cutflow"].to_boost().title == "dijethad"
        # assert f["cutflow"].to_boost().axes[0].name == "xaxis"
        # assert f["cutflow"].to_boost().axes[0].title == ""


def test_issue_0722():
    boost_histogram = pytest.importorskip("boost_histogram")

    with uproot.open(skhep_testdata.data_path("uproot-issue-722.root")) as f:
        f["hist"].to_boost()
