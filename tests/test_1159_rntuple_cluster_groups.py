# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import skhep_testdata

import uproot


def test_multiple_cluster_groups():
    filename = skhep_testdata.data_path(
        "test_multiple_cluster_groups_rntuple_v1-0-0-0.root"
    )
    with uproot.open(filename) as f:
        obj = f["ntuple"]

        group_ids = [cluster.cluster_group_id for cluster in obj.clusters]
        assert group_ids == [0] * 5 + [1] * 4 + [2] * 3

        assert obj.num_entries == 1000

        arrays = obj.arrays()

        assert arrays.one.tolist() == list(range(1000))
        assert arrays.int_vector.tolist() == [[i, i + 1] for i in range(1000)]
