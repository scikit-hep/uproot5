# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot

import numpy

try:
    import cupy
except ImportError:
    cupy = None

ak = pytest.importorskip("awkward")


@pytest.mark.parametrize(
    "backend,GDS,library",
    [
        ("cuda", False, cupy),
        pytest.param(
            "cuda",
            True,
            cupy,
            marks=pytest.mark.skipif(
                cupy is None, reason="could not import 'cupy': No module named 'cupy'"
            ),
        ),
    ],
)
def test_array_methods(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")
    filename = skhep_testdata.data_path(
        "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["Events"]
        nMuon_array = obj["nMuon"].array(backend=backend, use_GDS=GDS)
        Muon_pt_array = obj["Muon_pt"].array(backend=backend, use_GDS=GDS)
        assert ak.all(nMuon_array == library.array([len(l) for l in Muon_pt_array]))

        nMuon_arrays = obj["nMuon"].arrays(backend=backend, use_GDS=GDS)
        assert len(nMuon_arrays.fields) == 1
        assert len(nMuon_arrays) == 1000
        assert ak.all(nMuon_arrays["nMuon"] == nMuon_array)