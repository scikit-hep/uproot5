# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed`` opens only the first file that has the ``TTree`` (``open_files=True`` opens
every file), and ``uproot.root_write`` writes ROOT parts beside a ``graphed.aggregate_plan``'s
reductions from the same read."""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest

import uproot

pytest.importorskip("graphed.awkward")
pytest.importorskip("graphed_executors.local")

import graphed  # noqa: E402
import graphed.write  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402


def _make_tree(path, n=10, name="events"):
    with uproot.recreate(path) as f:
        f.mktree(name, {"x": "f8", "y": "f8"})
        f[name].extend({"x": np.arange(n, dtype="f8"), "y": 2.0 * np.arange(n)})
    return path


def _sum_plan(out, **kwargs):
    return graphed.aggregate_plan(
        out,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        **kwargs,
    )


def _run(plan):
    return SequentialRunner().run(plan).value


def _source_of(array):
    ((_nid, src),) = array.session.sources().items()
    return src


def _opens(monkeypatch):
    opened = []
    real_open = uproot.reading.ReadOnlyFile.__init__

    def counting(self, *args, **kwargs):
        opened.append(args[0])
        real_open(self, *args, **kwargs)

    monkeypatch.setattr(uproot.reading.ReadOnlyFile, "__init__", counting)
    return opened


def _reads(monkeypatch, array):
    reads = []
    source = _source_of(array)
    real = source.read_range

    def counting(tree, keys, start, stop):
        reads.append((tree.file.file_path, start, stop))
        return real(tree, keys, start, stop)

    monkeypatch.setattr(source, "read_range", counting)
    return reads


def _keys(array):
    return _source_of(array)._file_tree


# ---- lazy open -----------------------------------------------------------------------------
def test_only_the_first_file_with_the_tree_is_opened(tmp_path, monkeypatch):
    nott = _make_tree(os.path.join(tmp_path, "nott.root"), name="other")
    a, b, c = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "abc")
    files = [f + ":events" for f in (nott, a, b, c)]
    opened = _opens(monkeypatch)
    lazy = uproot.graphed(files, allow_missing=True)
    assert opened == [nott, a]
    assert [p.uri for p in _source_of(lazy).partitions(1)] == [a, b, c]
    opened.clear()
    eager = uproot.graphed(files, allow_missing=True, open_files=True)
    assert opened == [nott, a, b, c]
    assert [p.uri for p in _source_of(eager).partitions(1)] == [a, b, c]
    opened.clear()
    assert _run(_sum_plan(lazy.x)) == 135.0
    assert sorted(opened) == [a, b, c]  # each file is opened when it is read
    with pytest.raises(ValueError, match="no TTrees found in\n\n    {'.*nott.root'"):
        uproot.graphed(files[:1], allow_missing=True)
    with pytest.raises(TypeError, match="known_base_form= opens no file"):
        uproot.graphed(
            files[1:],
            known_base_form=lazy.session.form(lazy).tt.layout.form,
            open_files=True,
        )


def test_a_branch_missing_from_a_later_file_fails_its_read(tmp_path):
    a = _make_tree(os.path.join(tmp_path, "a.root"))
    b = os.path.join(tmp_path, "b.root")
    with uproot.recreate(b) as f:
        f.mktree("events", {"x": "f8"})
        f["events"].extend({"x": np.arange(3, dtype="f8")})
    lazy = uproot.graphed([a + ":events", b + ":events"])
    assert lazy.session.form(lazy).describe() == "## * {x: float64, y: float64}"
    assert _run(_sum_plan(lazy.x)) == 48.0
    with pytest.raises(uproot.KeyInFileError, match="y"):
        _run(_sum_plan(lazy.y))
    eager = uproot.graphed([a + ":events", b + ":events"], open_files=True)
    assert eager.session.form(eager).describe() == "## * {x: float64}"


def test_unopened_files_resolve_a_leaf_branch_and_a_duplicate_like_the_opened_one(
    tmp_path,
):
    a, b = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "ab")
    leaf = uproot.graphed([a + ":events/x", b + ":events/x"])
    assert _keys(leaf) == [(a, "/events;1"), (b, "events")]
    assert _run(_sum_plan(leaf.x)) == 90.0
    assert _keys(uproot.graphed([a, b, a])) == [
        (a, "/events;1"),
        (b, None),
        (a, "/events;1"),
    ]


def test_explicit_steps_of_unopened_files_are_their_partitions(tmp_path):
    a, b = (_make_tree(os.path.join(tmp_path, f"{x}.root")) for x in "ab")
    g = uproot.graphed(
        {
            a: {"object_path": "events", "steps": [[0, 4], [4, 10]]},
            b: {"object_path": "events", "steps": [[2, 5]]},
        }
    )
    assert [
        (p.uri, p.entry_start, p.entry_stop) for p in _source_of(g).partitions(1)
    ] == [(a, 0, 4), (a, 4, 10), (b, 2, 5)]
    assert _run(_sum_plan(g.x)) == 45.0 + 9.0


# ---- root_write ----------------------------------------------------------------------------
needs_part_write = pytest.mark.skipif(
    not hasattr(graphed.write, "PartWrite"), reason="root_write needs graphed 0.0.7"
)


def _trees(paths, tree="ev"):
    return [uproot.open(p)[tree].arrays() for p in paths]


def _step_name(partition):
    return f"part-{partition.blind_step:05d}.root"


@needs_part_write
def test_a_root_write_beside_a_reduction_reads_once_and_matches_graphed_write(
    tmp_path, monkeypatch
):
    src = _make_tree(os.path.join(tmp_path, "in.root"), n=20) + ":events"
    g = uproot.graphed(src)
    rec = gak.zip({"x": g.x, "y2": g.y * 2})
    reads = _reads(monkeypatch, g)
    plan = graphed.aggregate_plan(
        gak.sum(g.x),
        reduce=lambda vals: (float(vals[0]), [vals[1]]),
        combine=lambda a, b: (a[0] + b[0], a[1] + b[1]),
        empty=lambda: (0.0, []),
        writes=[
            uproot.root_write(
                rec, str(tmp_path / "composed"), name=_step_name, tree_name="ev"
            )
        ],
        steps_per_file=2,
    )
    total, composed = _run(plan)
    assert total == 190.0
    assert len(reads) == 2  # one read per partition serves the sum and the write
    reads.clear()
    standalone = uproot.graphed_write(
        rec,
        str(tmp_path / "alone"),
        steps_per_file=2,
        tree_name="ev",
        executor="thread",
    )
    assert len(reads) == 2
    assert [os.path.basename(p) for p in composed] == [
        os.path.basename(p) for p in standalone
    ]
    for got, want in zip(_trees(composed), _trees(standalone), strict=True):
        assert got.fields == want.fields == ["x", "y2"]
        assert got.tolist() == want.tolist()
    eager = uproot.open(src).arrays()
    assert (
        ak.concatenate(_trees(composed)).tolist()
        == ak.zip({"x": eager.x, "y2": eager.y * 2}).tolist()
    )


@needs_part_write
def test_every_task_of_a_root_write_writes_a_part_an_empty_one_included(tmp_path):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root"), n=3))
    plan = graphed.aggregate_plan(
        reduce=lambda vals: list(vals),
        combine=lambda a, b: a + b,
        empty=list,
        writes=[
            uproot.root_write(
                gak.zip({"x": g.x}), str(tmp_path / "out"), name=_step_name
            )
        ],
        steps_per_file=5,
    )
    paths = _run(plan)
    assert [len(t) for t in _trees(paths, "tree")] == [0, 1, 0, 1, 1]


@needs_part_write
def test_root_write_refusals(tmp_path, monkeypatch):
    g = uproot.graphed(_make_tree(os.path.join(tmp_path, "in.root")))
    with pytest.raises(TypeError, match="root_write writes a record's fields"):
        uproot.root_write(g.x, str(tmp_path), name=_step_name)
    monkeypatch.setattr(graphed, "__version__", "0.0.6")
    with pytest.raises(ModuleNotFoundError, match="needs graphed 0.0.7 or newer"):
        uproot.root_write(gak.zip({"x": g.x}), str(tmp_path), name=_step_name)
