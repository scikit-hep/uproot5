# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

import uproot

x = ak.Array([[1.1], [2.2, 3.3], [4.4, 5.5, 6.6]])
jet_pt = ak.Array([[10.0], [20.0, 30.0], []])
jet_eta = ak.Array([[0.1], [0.2, 0.3], []])


def counter_name(counted):
    return "nJet" if counted.startswith("Jet_") else "n" + counted


def make_tree(f):
    # 'nx' is declared before the jagged 'x' whose counter replaces it, and
    # 'Jet_pt' and 'Jet_eta' share the counter 'nJet'
    return f.mktree(
        "t",
        {
            "nx": np.int32,
            "y": np.float64,
            "x": x.type,
            "Jet_pt": jet_pt.type,
            "Jet_eta": jet_eta.type,
        },
        counter_name=counter_name,
    )


def test_counter_collisions(tmp_path):
    path = tmp_path / "file.root"
    y = np.array([10.0, 20.0, 30.0])

    with uproot.recreate(path) as f:
        tree = make_tree(f)

        lookup = tree._cascading._branch_lookup
        data = tree._cascading._branch_data
        assert list(lookup) == ["nx", "y", "x", "nJet", "Jet_pt", "Jet_eta"]
        for name, index in lookup.items():
            assert data[index]["fName"] == name
        assert data[lookup["nx"]]["kind"] == "counter"
        assert data[lookup["Jet_pt"]]["counter"] is data[lookup["nJet"]]
        assert data[lookup["Jet_eta"]]["counter"] is data[lookup["nJet"]]

        tree.extend(
            {"nx": [1, 2, 3], "y": y, "x": x, "Jet_pt": jet_pt, "Jet_eta": jet_eta}
        )

    with uproot.open(path) as f:
        result = f["t"].arrays()
        assert result["nx"].tolist() == [1, 2, 3]
        assert result["y"].tolist() == y.tolist()
        assert result["x"].tolist() == x.tolist()
        assert result["nJet"].tolist() == [1, 2, 0]
        assert result["Jet_pt"].tolist() == jet_pt.tolist()
        assert result["Jet_eta"].tolist() == jet_eta.tolist()


@pytest.mark.parametrize(
    "overrides",
    [
        {"nx": [9, 9, 9]},
        {"Jet_eta": ak.Array([[0.1], [0.2], [0.3]])},
    ],
)
def test_counter_disagreement(tmp_path, overrides):
    data = {
        "nx": [1, 2, 3],
        "y": [10.0, 20.0, 30.0],
        "x": x,
        "Jet_pt": jet_pt,
        "Jet_eta": jet_eta,
    }
    with uproot.recreate(tmp_path / "file.root") as f:
        tree = make_tree(f)
        with pytest.raises(ValueError, match="disagree"):
            tree.extend({**data, **overrides})


def concat_fields(outer, inner):
    return outer + inner


record = ak.Array([{"a": 1}]).type
jagged_record = ak.Array([[{"n": 1}]]).type


@pytest.mark.parametrize(
    ("branch_types", "field_name"),
    [
        ({"nx": np.float64, "x": x.type}, None),
        ({"nx": np.dtype(("i4", (3,))), "x": x.type}, None),
        ({"nx": x.type, "x": x.type}, None),
        ({"nx": {"a": np.int32}, "x": x.type}, None),
        ({"x": x.type, "nx": {"a": np.int32}}, None),
        ({"x": x.type, "nx": record}, None),
        ({"n": {"x": np.int32}, "x": x.type}, concat_fields),
        ({"x": x.type, "n": {"x": np.int32}}, concat_fields),
        ({"x": x.type, "n": ak.Array([{"x": 1}]).type}, concat_fields),
        ({"n": jagged_record}, concat_fields),
    ],
    ids=[
        "float",
        "subarray",
        "jagged",
        "record-before",
        "record-after",
        "awkward-record-after",
        "record-field-before",
        "record-field-after",
        "awkward-record-field-after",
        "own-jagged-record-field",
    ],
)
def test_counter_collides_with_incompatible_branch(tmp_path, branch_types, field_name):
    kwargs = {} if field_name is None else {"field_name": field_name}
    with uproot.recreate(tmp_path / "file.root") as f:
        with pytest.raises(ValueError, match="collides"):
            f.mktree("t", branch_types, **kwargs)
