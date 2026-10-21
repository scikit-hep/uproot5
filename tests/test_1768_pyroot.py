# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import os
import subprocess
import sys

import numpy as np
import pytest

import uproot

ROOT = pytest.importorskip("ROOT")


DIMENSIONS = {
    "TH1D": 1,
    "TH1F": 1,
    "TH1I": 1,
    "TH2D": 2,
    "TH3D": 3,
    "TProfile": 1,
    "TProfile2D": 2,
    "TProfile3D": 3,
}


def _make(kind, name, weighted):
    nd = DIMENSIONS[kind]
    is_profile = kind.startswith("TProfile")
    bins = [(5, 0.0, 1.0), (4, 0.0, 1.0), (3, 0.0, 1.0)][:nd]
    obj = getattr(ROOT, kind)(name, f"title of {name}", *sum(bins, ()))
    if kind == "TH1D" and not weighted:
        for i, label in enumerate("abcde"):
            obj.GetXaxis().SetBinLabel(i + 1, label)
    if weighted:
        obj.Sumw2()

    rng = np.random.default_rng(12345)
    for _ in range(200):
        args = list(rng.uniform(-0.2, 1.2, nd + is_profile))
        if weighted:
            args.append(rng.uniform(0.5, 2.0))
        obj.Fill(*args)
    return obj


def _assert_same(expected, actual):
    assert actual.ClassName() == expected.ClassName()
    assert actual.GetName() == expected.GetName()
    assert actual.GetTitle() == expected.GetTitle()
    assert actual.GetEntries() == expected.GetEntries()
    assert actual.GetSumw2N() == expected.GetSumw2N()

    stats_expected, stats_actual = np.zeros(20), np.zeros(20)
    expected.GetStats(stats_expected)
    actual.GetStats(stats_actual)
    assert stats_actual.tolist() == stats_expected.tolist()

    methods = ["GetBinContent", "GetBinError"]
    if hasattr(expected, "GetBinEntries"):
        methods += ["GetBinEntries", "GetBinEffectiveEntries"]
        assert actual.GetErrorOption() == expected.GetErrorOption()
    for method in methods:
        assert [getattr(actual, method)(i) for i in range(actual.GetNcells())] == [
            getattr(expected, method)(i) for i in range(expected.GetNcells())
        ]

    for axis in "XYZ":
        axis_expected = getattr(expected, f"Get{axis}axis")()
        axis_actual = getattr(actual, f"Get{axis}axis")()
        assert axis_actual.GetNbins() == axis_expected.GetNbins()
        assert axis_actual.GetXmin() == axis_expected.GetXmin()
        assert axis_actual.GetXmax() == axis_expected.GetXmax()
        assert [
            axis_actual.GetBinLabel(i) for i in range(1, axis_actual.GetNbins() + 1)
        ] == [
            axis_expected.GetBinLabel(i) for i in range(1, axis_expected.GetNbins() + 1)
        ]


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("kind", DIMENSIONS)
def test_read_from_root_file(tmp_path, kind, weighted):
    filename = os.path.join(tmp_path, "root_written.root")
    fout = ROOT.TFile(filename, "RECREATE")
    _make(kind, "obj", weighted).Write()
    fout.Close()

    fin = ROOT.TFile(filename)
    with uproot.open(filename) as file:
        _assert_same(fin.Get("obj"), uproot.to_pyroot(file["obj"]))
    fin.Close()


@pytest.mark.parametrize(
    ("ndim", "storage"),
    [
        (ndim, storage)
        for ndim in (1, 2, 3)
        for storage in ("Double", "Weight", "Mean", "WeightedMean")
        # Uproot can't convert 2D or 3D profiles from hist yet
        if ndim == 1 or "Mean" not in storage
    ],
)
def test_converted_from_hist(tmp_path, ndim, storage):
    hist = pytest.importorskip("hist")

    axes = [hist.axis.Regular(5, 0, 1), hist.axis.Variable([0, 0.2, 0.7, 1])]
    axes.append(hist.axis.Regular(3, 0, 1))
    h = hist.Hist(*axes[:ndim], storage=getattr(hist.storage, storage)())
    rng = np.random.default_rng(12345)
    kwargs = {}
    if storage in ("Weight", "WeightedMean"):
        kwargs["weight"] = rng.uniform(0.5, 2.0, 200)
    if storage in ("Mean", "WeightedMean"):
        kwargs["sample"] = rng.uniform(-1, 1, 200)
    h.fill(*rng.uniform(-0.2, 1.2, (ndim, 200)), **kwargs)

    filename = os.path.join(tmp_path, "uproot_written.root")
    with uproot.recreate(filename) as file:
        file["obj"] = h

    fin = ROOT.TFile(filename)
    _assert_same(fin.Get("obj"), uproot.to_pyroot(uproot.to_writable(h), "obj"))
    fin.Close()


GRAPH_ERRORS = {
    "TGraph": [],
    "TGraphErrors": ["x_errors", "y_errors"],
    "TGraphAsymmErrors": [
        "x_errors_low",
        "x_errors_high",
        "y_errors_low",
        "y_errors_high",
    ],
}
GRAPH_GETTERS = {
    "x": "GetX",
    "y": "GetY",
    "x_errors": "GetEX",
    "y_errors": "GetEY",
    "x_errors_low": "GetEXlow",
    "x_errors_high": "GetEXhigh",
    "y_errors_low": "GetEYlow",
    "y_errors_high": "GetEYhigh",
}


def _assert_same_graph(expected, actual):
    assert actual.ClassName() == expected.ClassName()
    assert actual.GetName() == expected.GetName()
    assert actual.GetTitle() == expected.GetTitle()
    assert actual.GetMinimum() == expected.GetMinimum()
    assert actual.GetMaximum() == expected.GetMaximum()
    n = expected.GetN()
    assert actual.GetN() == n
    for method in GRAPH_GETTERS.values():
        values_expected = getattr(expected, method)()
        values_actual = getattr(actual, method)()
        if values_expected:
            assert list(values_actual)[:n] == list(values_expected)[:n]
        else:
            assert not values_actual


@pytest.mark.parametrize("kind", GRAPH_ERRORS)
def test_graph(tmp_path, kind):
    rng = np.random.default_rng(12345)
    columns = {name: rng.uniform(0, 1, 7) for name in ["x", "y", *GRAPH_ERRORS[kind]]}

    root_filename = os.path.join(tmp_path, "root_written.root")
    fout = ROOT.TFile(root_filename, "RECREATE")
    graph = getattr(ROOT, kind)(7, *columns.values())
    graph.SetNameTitle("graph", "title of graph")
    graph.Write()
    fout.Close()

    uproot_filename = os.path.join(tmp_path, "uproot_written.root")
    with uproot.recreate(uproot_filename) as file:
        file["graph"] = uproot.as_TGraph(columns, title="title of graph")

    fin = ROOT.TFile(uproot_filename)
    graph = fin.Get("graph")
    for column, values in columns.items():
        assert list(getattr(graph, GRAPH_GETTERS[column])())[:7] == values.tolist()
    fin.Close()

    for filename in [root_filename, uproot_filename]:
        fin = ROOT.TFile(filename)
        with uproot.open(filename) as file:
            _assert_same_graph(fin.Get("graph"), uproot.to_pyroot(file["graph"]))
        fin.Close()

    fin = ROOT.TFile(uproot_filename)
    _assert_same_graph(
        fin.Get("graph"),
        uproot.to_pyroot(uproot.as_TGraph(columns, title="title of graph"), "graph"),
    )
    fin.Close()


def test_streamer_info_registration():
    # A TMessage carries no StreamerInfo, so ROOT can't read the TGraph version
    # that Uproot writes unless to_pyroot registers it. Check this in a new
    # process, since opening any file written by Uproot would also register it.
    script = (
        "import uproot\n"
        "data = {'x': [1.0, 2.0], 'y': [3.0, 4.0]}\n"
        "graph = uproot.to_pyroot(uproot.as_TGraph(data), 'g')\n"
        "print(graph.GetN(), graph.GetName(), list(graph.GetY()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.stdout.splitlines()[-1] == "2 g [3.0, 4.0]"
