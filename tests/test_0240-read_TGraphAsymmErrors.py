# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE


import numpy
import pytest

import uproot

requests = pytest.importorskip("requests")


@pytest.mark.network
@pytest.fixture(scope="module")
def datafile(tmpdir_factory):
    response = requests.get(
        "https://www.hepdata.net/download/table/ins1755298/Expected%20limit%201lbb/3/root"
    )
    response.raise_for_status()
    tmp_file = tmpdir_factory.mktemp("data").join(
        "HEPData-ins1755298-v3-Expected_limit_1lbb.root"
    )
    with open(str(tmp_file), "wb") as f:
        f.write(response.content)
    yield str(tmp_file)


@pytest.fixture
def graph(datafile):
    with uproot.open(datafile) as f:
        yield f["Expected limit 1lbb/Graph1D_y1"]


@pytest.mark.skip(
    reason="HEPData can't supply this file: https://www.hepdata.net/download/table/ins1755298/Expected%20limit%201lbb/3/root"
)
def test_interpretation(graph):
    assert graph.classname == "TGraphAsymmErrors"
    assert graph.behaviors[0] == uproot.behaviors.TGraphAsymmErrors.TGraphAsymmErrors

    assert "fX" in graph.all_members.keys()
    assert "fY" in graph.all_members.keys()
    assert "fEXlow" in graph.all_members.keys()
    assert "fEYlow" in graph.all_members.keys()
    assert "fEXhigh" in graph.all_members.keys()
    assert "fEYhigh" in graph.all_members.keys()


@pytest.mark.skip(
    reason="HEPData can't supply this file: https://www.hepdata.net/download/table/ins1755298/Expected%20limit%201lbb/3/root"
)
@pytest.mark.parametrize("axis", [-2, -1, 0, 1, "x", "y"])
def test_values_single(graph, axis):
    values = graph.values(axis=axis)
    assert isinstance(values, numpy.ndarray)
    assert values.shape == (162,)


@pytest.mark.skip(
    reason="HEPData can't supply this file: https://www.hepdata.net/download/table/ins1755298/Expected%20limit%201lbb/3/root"
)
@pytest.mark.parametrize("axis", [-2, -1, 0, 1, "x", "y"])
@pytest.mark.parametrize("which", ["low", "high", "mean", "diff"])
def test_errors_single(graph, axis, which):
    errors = graph.errors(axis=axis, which=which)
    assert isinstance(errors, numpy.ndarray)
    assert errors.shape == (162,)


@pytest.mark.skip(
    reason="HEPData can't supply this file: https://www.hepdata.net/download/table/ins1755298/Expected%20limit%201lbb/3/root"
)
@pytest.mark.parametrize("axis", ["both"])
def test_values_double(graph, axis):
    values = graph.values(axis=axis)
    assert len(values) == 2
    assert all(isinstance(arr, numpy.ndarray) for arr in values)
    assert all(arr.shape == (162,) for arr in values)


@pytest.mark.skip(
    reason="HEPData can't supply this file: https://www.hepdata.net/download/table/ins1755298/Expected%20limit%201lbb/3/root"
)
@pytest.mark.parametrize("axis", ["both"])
@pytest.mark.parametrize("which", ["low", "high", "mean", "diff"])
def test_errors_double(graph, axis, which):
    errors = graph.errors(axis=axis, which=which)
    assert len(errors) == 2
    assert all(isinstance(arr, numpy.ndarray) for arr in errors)
    assert all(arr.shape == (162,) for arr in errors)
