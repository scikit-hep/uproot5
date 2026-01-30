# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest
import skhep_testdata

import uproot

ak = pytest.importorskip("awkward")
cupy = pytest.importorskip("cupy")
pytestmark = [
    pytest.mark.skipif(
        cupy.cuda.runtime.driverGetVersion() == 0, reason="No available CUDA driver."
    ),
    pytest.mark.xfail(
        strict=False,
        reason="There are breaking changes in new versions of KvikIO that are not yet resolved",
    ),
]


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_array_methods(backend, interpreter, library):
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        nMuon_array = obj["nMuon"].array(backend=backend, interpreter=interpreter)
        Muon_pt_array = obj["Muon_pt"].array(backend=backend, interpreter=interpreter)
        assert ak.all(nMuon_array == library.array([len(l) for l in Muon_pt_array]))

        nMuon_arrays = obj["nMuon"].arrays(backend=backend, interpreter=interpreter)
        assert len(nMuon_arrays.fields) == 1
        assert len(nMuon_arrays) == 1000
        assert ak.all(nMuon_arrays["nMuon"] == nMuon_array)


@pytest.mark.parametrize(
    ("backend", "interpreter", "library"),
    [("cuda", "cpu", cupy), ("cuda", "gpu", cupy)],
)
def test_iterate(backend, interpreter, library):
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        for i, arrays in enumerate(
            obj.iterate(step_size=100, backend=backend, interpreter=interpreter)
        ):
            assert len(arrays) == 100
            if i == 0:
                expected_pt = [10.763696670532227, 15.736522674560547]
                expected_charge = [-1, -1]
                assert arrays["Muon_pt"][0].tolist() == expected_pt
                assert arrays["Muon_charge"][0].tolist() == expected_charge

        for i, arrays in enumerate(
            obj.iterate(step_size="10 kB", backend=backend, interpreter=interpreter)
        ):
            if i == 0:
                assert len(arrays) == 384
                expected_pt = [10.763696670532227, 15.736522674560547]
                expected_charge = [-1, -1]
                assert arrays["Muon_pt"][0].tolist() == expected_pt
                assert arrays["Muon_charge"][0].tolist() == expected_charge
            elif i == 1:
                assert len(arrays) == 384
            elif i == 2:
                assert len(arrays) == 232
            else:
                assert False

        Muon_pt = obj["Muon_pt"]
        for i, arrays in enumerate(
            Muon_pt.iterate(step_size=100, backend=backend, interpreter=interpreter)
        ):
            assert len(arrays) == 100
            if i == 0:
                expected_pt = [10.763696670532227, 15.736522674560547]
                assert arrays["Muon_pt"][0].tolist() == expected_pt

        for i, arrays in enumerate(
            Muon_pt.iterate(step_size="5 kB", backend=backend, interpreter=interpreter)
        ):
            if i == 0:
                assert len(arrays) == 611
                expected_pt = [10.763696670532227, 15.736522674560547]
                assert arrays["Muon_pt"][0].tolist() == expected_pt
            elif i == 1:
                assert len(arrays) == 389
            else:
                assert False
