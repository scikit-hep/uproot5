# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import pytest
import skhep_testdata

try:
    import cupy
except ImportError:
    cupy = None
import uproot

pytest.importorskip("awkward")


@pytest.mark.parametrize(
    ("backend", "GDS", "library"),
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
def test_flat(backend, GDS, library):
    if GDS and cupy.cuda.runtime.driverGetVersion() == 0:
        pytest.skip("No available CUDA driver.")

    filename = skhep_testdata.data_path("test_int_float_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert R.keys() == ["one_integers", "two_floats"]
        assert [r.type_name for r in R.header.field_records] == [
            "std::int32_t",
            "float",
        ]
        assert R.header.checksum == R.footer.header_checksum
        assert all(
            R.arrays(entry_stop=3, use_GDS=GDS, backend=backend)["one_integers"]
            == library.array([9, 8, 7])
        )
        assert all(
            R.arrays("one_integers", entry_stop=3, use_GDS=GDS, backend=backend)[
                "one_integers"
            ]
            == library.array([9, 8, 7])
        )
        assert all(
            R.arrays(entry_start=1, entry_stop=3, use_GDS=GDS, backend=backend)[
                "one_integers"
            ]
            == library.array([8, 7])
        )

    filename = skhep_testdata.data_path("test_int_5e4_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        R = f["ntuple"]
        assert all(
            R.arrays(entry_stop=3, use_GDS=GDS, backend=backend)["one_integers"]
            == library.array([50000, 49999, 49998])
        )
        assert all(
            R.arrays(entry_start=-3, use_GDS=GDS, backend=backend)["one_integers"]
            == library.array([3, 2, 1])
        )
