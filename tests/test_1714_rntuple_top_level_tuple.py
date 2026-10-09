# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak

import uproot


def test_top_level_tuple_round_trip(tmp_path):
    array = ak.Array([(1.0, 2), (3.0, 4)])
    path = tmp_path / "tuple.root"

    with uproot.recreate(path) as file:
        file["tuple"] = array

    with uproot.open(path) as file:
        ntuple = file["tuple"]
        form, _ = ntuple.to_akform()

        assert form.is_tuple
        assert ntuple.field_names == ["_0", "_1"]
        assert ntuple.arrays().tolist() == array.tolist()
