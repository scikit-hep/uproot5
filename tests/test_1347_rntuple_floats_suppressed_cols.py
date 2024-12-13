# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata
import numpy as np

import uproot


def test_custom_floats():
    filename = skhep_testdata.data_path("test_float_types_rntuple_v1-0-0-0.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        arrays = obj.arrays()

        entry = arrays[0]
        true_value = 1.23456789
        assert np.isclose(entry.trunc10, true_value, rtol=0.25)
        assert np.isclose(entry.trunc16, true_value, rtol=1e-3)
        assert np.isclose(entry.trunc24, true_value)
        assert np.isclose(entry.trunc31, true_value)
        assert np.isclose(entry.quant1, true_value, rtol=2)
        assert np.isclose(entry.quant8, true_value, rtol=1e-3)
        assert np.isclose(entry.quant16, true_value, rtol=1e-4)
        assert np.isclose(entry.quant20, true_value)
        assert np.isclose(entry.quant24, true_value)
        assert np.isclose(entry.quant25, true_value)
        assert np.isclose(entry.quant32, true_value)

        entry = arrays[1]
        true_value = 1.4660155e13
        assert np.isclose(entry.trunc10, true_value, rtol=0.25)
        assert np.isclose(entry.trunc16, true_value, rtol=1e-2)
        assert np.isclose(entry.trunc24, true_value)
        assert np.isclose(entry.trunc31, true_value)
        true_value = 1.6666666
        assert np.isclose(entry.quant1, true_value, rtol=2)
        assert np.isclose(entry.quant8, true_value, rtol=1e-3)
        assert np.isclose(entry.quant16, true_value, rtol=1e-4)
        assert np.isclose(entry.quant20, true_value)
        assert np.isclose(entry.quant24, true_value)
        assert np.isclose(entry.quant25, true_value)
        assert np.isclose(entry.quant32, true_value)

        entry = arrays[2]
        true_value = -6.2875986e-22
        assert np.isclose(entry.trunc10, true_value, rtol=0.25)
        assert np.isclose(entry.trunc16, true_value, rtol=1e-3)
        assert np.isclose(entry.trunc24, true_value)
        assert np.isclose(entry.trunc31, true_value)
        assert np.isclose(entry.quant1, true_value, atol=2.1)
        assert np.isclose(entry.quant8, true_value, rtol=1e-3)
        assert np.isclose(entry.quant16, true_value, rtol=1e-4)
        assert np.isclose(entry.quant20, true_value)
        assert np.isclose(entry.quant24, true_value)
        assert np.isclose(entry.quant25, true_value, atol=1e-6)
        assert np.isclose(entry.quant32, true_value)

        entry = arrays[3]
        true_value = -1.9060668
        assert np.isclose(entry.trunc10, true_value, rtol=0.25)
        assert np.isclose(entry.trunc16, true_value, rtol=1e-2)
        assert np.isclose(entry.trunc24, true_value, rtol=1e-3)
        assert np.isclose(entry.trunc31, true_value)
        assert np.isclose(entry.quant1, true_value, rtol=2)
        assert np.isclose(entry.quant8, true_value, rtol=1e-2)
        assert np.isclose(entry.quant16, true_value, rtol=1e-4)
        assert np.isclose(entry.quant20, true_value)
        assert np.isclose(entry.quant24, true_value)
        assert np.isclose(entry.quant25, true_value)
        assert np.isclose(entry.quant32, true_value)


def test_multiple_representations():
    filename = skhep_testdata.data_path(
        "test_multiple_representations_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        assert len(obj.page_list_envelopes.pagelinklist) == 3
        # The zeroth representation is active in clusters 0 and 2, but not in cluster 1
        assert not obj.page_list_envelopes.pagelinklist[0][0].suppressed
        assert obj.page_list_envelopes.pagelinklist[1][0].suppressed
        assert not obj.page_list_envelopes.pagelinklist[2][0].suppressed

        arrays = obj.arrays()

        assert np.allclose(arrays.real, [1, 2, 3])
