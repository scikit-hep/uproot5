# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot

ak = pytest.importorskip("awkward")

_typenames = {
    np.float64: "double",
    np.float32: "float",
    np.int32: "int32_t",
    np.uint16: "uint16_t",
}


def _jagged(rng, counts, inner_shape, dtype):
    total = int(np.sum(counts))
    flat = (rng.random((total, *inner_shape)) * 100).astype(dtype)
    return ak.unflatten(flat, counts)


def _batches(inner_shape, dtype):
    rng = np.random.default_rng(1490)
    return [
        _jagged(rng, [0, 3, 1, 0, 2], inner_shape, dtype),
        _jagged(rng, [0, 0, 0], inner_shape, dtype),  # only empty lists
        _jagged(rng, [5, 0, 1, 4], inner_shape, dtype),
    ]


@pytest.mark.parametrize("inner_shape", [(1,), (3,), (6,), (2, 3)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32, np.uint16])
def test_round_trip(tmp_path, inner_shape, dtype):
    path = str(tmp_path / "test.root")
    batches = _batches(inner_shape, dtype)

    with uproot.recreate(path) as f:
        tree = f.mktree("tree", {"x": batches[0].type.content})
        for batch in batches:
            tree.extend({"x": batch})

    expected = ak.concatenate(batches)
    with uproot.open(path) as f:
        branch = f["tree"]["x"]
        assert branch.num_baskets == len(batches)
        assert branch.typename == _typenames[dtype] + "[]" + "".join(
            f"[{n}]" for n in inner_shape
        )
        assert branch.member("fLeaves")[0].member("fLen") == int(np.prod(inner_shape))
        array = branch.array()
        assert str(array.type.content) == str(expected.type.content)
        assert array.tolist() == expected.tolist()
        assert f["tree"]["nx"].array(library="np").tolist() == ak.num(expected).tolist()


def test_mktree_with_data(tmp_path):
    # the reproducer from the issue: var * 6 * float64, written in one basket
    path = str(tmp_path / "test.root")
    wfm = ak.from_regular(
        ak.Array(np.arange(1000 * 5 * 6, dtype=np.float64).reshape(1000, 5, 6)),
        axis=1,
    )
    assert str(wfm.type) == "1000 * var * 6 * float64"

    with uproot.recreate(path) as f:
        f.mktree("tree", {"wfm": wfm})

    with uproot.open(path) as f:
        assert f["tree"]["wfm"].num_baskets == 1
        assert f["tree"]["wfm"].array().tolist() == wfm.tolist()


def test_unusual_layouts(tmp_path):
    path = str(tmp_path / "test.root")
    flat = np.arange(10 * 6, dtype=np.float64).reshape(10, 6)
    counts = [3, 0, 2, 5]

    # inner dimension as a NumpyArray with inner shape
    numpy_inner = ak.unflatten(ak.from_numpy(flat, regulararray=False), counts)
    # inner dimension as a RegularArray, sliced so that the offsets don't start at 0
    sliced = ak.unflatten(ak.from_numpy(flat, regulararray=True), counts)[2:]
    # ListArray instead of ListOffsetArray
    listarray = numpy_inner[[3, 0, 2]]
    # RegularArray whose content is longer than it needs to be
    overlong = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3, 3, 5, 10])),
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.arange(10 * 6 + 4, dtype=np.float64)), 6
            ),
        )
    )
    batches = [numpy_inner, sliced, listarray, overlong]

    with uproot.recreate(path) as f:
        tree = f.mktree("tree", {"x": "var * 6 * float64"})
        for batch in batches:
            tree.extend({"x": batch})

    with uproot.open(path) as f:
        assert f["tree"]["x"].array().tolist() == ak.concatenate(batches).tolist()


def test_update_and_extend(tmp_path):
    path = str(tmp_path / "test.root")
    batches_a = _batches((6,), np.float64)
    batches_b = _batches((2, 3), np.int32)

    with uproot.recreate(path) as f:
        f.mktree("tree", {"a": batches_a[0], "b": batches_b[0]})

    for a, b in zip(batches_a[1:], batches_b[1:]):
        with uproot.update(path) as f:
            f["tree"].extend({"a": a, "b": b})

    with uproot.open(path) as f:
        arrays = f["tree"].arrays(["a", "b"])
        assert arrays.a.tolist() == ak.concatenate(batches_a).tolist()
        assert arrays.b.tolist() == ak.concatenate(batches_b).tolist()


def test_read_with_ROOT(tmp_path, capfd):
    ROOT = pytest.importorskip("ROOT")

    path = str(tmp_path / "test.root")
    branches = {
        "d6": _batches((6,), np.float64),
        "f3": _batches((3,), np.float32),
        "i23": _batches((2, 3), np.int32),
    }

    with uproot.recreate(path) as f:
        tree = f.mktree(
            "tree", {name: b[0].type.content for name, b in branches.items()}
        )
        for i in range(3):
            tree.extend({name: b[i] for name, b in branches.items()})

    expected = {name: ak.concatenate(b).tolist() for name, b in branches.items()}

    file = ROOT.TFile.Open(path)
    tree = file.Get("tree")
    assert tree.GetEntries() == len(expected["d6"])
    for name in branches:
        leaf = tree.GetLeaf(name)
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            values = [leaf.GetValue(j) for j in range(leaf.GetLen())]
            assert values == pytest.approx(np.ravel(expected[name][i]).tolist())
    file.Close()

    captured = capfd.readouterr()
    assert "repair" not in captured.out + captured.err
