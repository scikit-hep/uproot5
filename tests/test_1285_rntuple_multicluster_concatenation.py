# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import skhep_testdata
import numpy as np

import uproot


def test_schema_extension():
    filename = skhep_testdata.data_path("test_ntuple_index_multicluster.root")
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        arrays = obj.arrays()
        int_vec_array = arrays["int_vector"]

        true_array = np.zeros((200, 2), dtype=np.int16)
        true_array[:100, 0] = np.arange(100, dtype=np.int16)
        true_array[:100, 1] = np.arange(100, dtype=np.int16)
        true_array[100:, 0] = np.arange(100, dtype=np.int16)
        true_array[100:, 1] = np.arange(100, dtype=np.int16) + 1

        assert np.array_equal(int_vec_array, true_array)
