# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Edge cases of the ``graphed`` entry points over real files.

- ``allow_missing=True`` skips a file that does not contain the requested object, both when
  recording an array and when partitioning a dataset; an empty or branch-less selection is
  rejected loudly instead of producing an empty array;
- ``resolve_read_branches`` falls back to the requested path where no counter branch can serve an
  ``OFFSETS`` need (``RNTuple`` fields, and paths the ``TTree`` does not have);
- partitioning a tree into more steps than it has entries emits no empty chunk;
- ``graphed_head`` reads exactly the branches the recorded graph accesses — a field subset reads
  those fields, a whole-record operation reads every selected branch — and refuses an array that
  is not backed by a single ``uproot.graphed`` source;
- with ``known_base_form=`` (no file opened) the source is named from the object path, whatever
  ``TDirectory`` prefix or ``;cycle`` suffix it carries, and when it names no object at all;
- the missing-dependency errors of ``uproot.extras`` carry their install hints.
"""

import os
import sys

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.extras

pytest.importorskip("graphed.awkward")

from graphed import BufferNeed  # noqa: E402
from graphed.awkward import AwkwardBackend, AwkwardForm, gak  # noqa: E402

from uproot._dask import _get_ttree_form  # noqa: E402
from uproot._graphed import _evaluation_columns, _GraphedTTreeSource  # noqa: E402


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _hzz():
    return skhep_testdata.data_path("uproot-HZZ.root") + ":events"


def _good_and_treeless(tmp_path):
    """A file holding ``events`` (an ``RNTuple``: dict assignment), and one without it."""
    good = os.path.join(tmp_path, "good.root")
    with uproot.recreate(good) as f:
        f["events"] = {"x": np.arange(6.0)}
    other = os.path.join(tmp_path, "other.root")
    with uproot.recreate(other) as f:
        f["something_else"] = {"x": np.arange(3.0)}
    return good, other


# ---- allow_missing --------------------------------------------------------------------------
def test_allow_missing_skips_a_file_without_the_tree(tmp_path):
    good, other = _good_and_treeless(tmp_path)
    files = [good + ":events", other + ":events"]
    g = uproot.graphed(files, library="ak", allow_missing=True)
    got = ak.Array(g.session.materialize(g.x * 2.0))
    assert got.to_list() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # the good file alone
    with pytest.raises(KeyError, match="events"):
        uproot.graphed(files, library="ak")


def test_allow_missing_skips_a_file_without_the_tree_when_partitioning(tmp_path):
    good, other = _good_and_treeless(tmp_path)
    files = [good + ":events", other + ":events"]
    tasks = uproot.graphed_partitions(files, allow_missing=True)
    assert [
        (t.partition.uri, t.partition.entry_start, t.partition.entry_stop)
        for t in tasks
    ] == [(good, 0, 6)]
    with pytest.raises(KeyError, match="events"):
        uproot.graphed_partitions(files)


def test_no_tree_in_any_file_is_rejected(tmp_path):
    _, other = _good_and_treeless(tmp_path)
    with pytest.raises(ValueError, match="no TTrees found"):
        uproot.graphed(other + ":events", library="ak", allow_missing=True)


def test_selection_without_a_common_branch_is_rejected(tmp_path):
    a = os.path.join(tmp_path, "a.root")
    with uproot.recreate(a) as f:
        f["events"] = {"x": np.arange(4.0)}
    b = os.path.join(tmp_path, "b.root")
    with uproot.recreate(b) as f:
        f["events"] = {"y": np.arange(4.0)}
    with pytest.raises(ValueError, match="no TBranches in common"):
        uproot.graphed([a + ":events", b + ":events"], library="ak")
    with pytest.raises(ValueError, match="no TBranches in common"):
        uproot.graphed(a + ":events", library="ak", filter_name="not_a_branch")


# ---- OFFSETS needs with no counter branch ---------------------------------------------------
def test_offsets_need_on_an_rntuple_reads_the_field_itself(tmp_path, monkeypatch):
    path = os.path.join(tmp_path, "rntuple.root")
    data = ak.Array({"jag": [[1.0, 2.0], [3.0], [], [4.0, 5.0, 6.0]]})
    with uproot.recreate(path) as f:
        f.mkrntuple("ntuple", data)
    obj = uproot.open(path)["ntuple"]

    g = uproot.graphed(path + ":ntuple", library="ak")
    counts = gak.num(g.jag, axis=1)
    assert uproot.necessary_buffers(counts) == {"ntuple": {"jag": BufferNeed.OFFSETS}}
    assert ak.Array(g.session.materialize(counts)).to_list() == [2, 1, 0, 3]

    def no_field_lookup(self, where):
        raise AssertionError(f"an RNTuple field was looked up for a counter: {where!r}")

    monkeypatch.setattr(type(obj), "__getitem__", no_field_lookup)
    # an RNTuple's index column is not addressable on its own: read the field
    assert uproot.resolve_read_branches(obj, {"jag": BufferNeed.OFFSETS}) == {
        "jag": "jag"
    }


def test_offsets_need_for_an_absent_branch_reads_the_requested_path():
    tree = uproot.open(_hzz())
    with pytest.raises(KeyError):
        tree["absent"]  # no branch, so no counter branch to look up either
    assert uproot.resolve_read_branches(tree, {"absent": BufferNeed.OFFSETS}) == {
        "absent": "absent"
    }
    # a branch that exists but carries no counter behaves the same way
    assert tree["MET_px"].count_branch is None
    assert uproot.resolve_read_branches(tree, {"MET_px": BufferNeed.OFFSETS}) == {
        "MET_px": "MET_px"
    }


# ---- partitioning ---------------------------------------------------------------------------
def test_more_steps_than_entries_emits_no_empty_partition(tmp_path):
    path = os.path.join(tmp_path, "tiny.root")
    with uproot.recreate(path) as f:
        f["events"] = {"x": np.arange(3.0)}
    tasks = uproot.graphed_partitions(path + ":events", steps_per_file=5)
    ranges = [(t.partition.entry_start, t.partition.entry_stop) for t in tasks]
    assert ranges == [(0, 1), (1, 2), (2, 3)]  # tiles [0, 3) exactly once, no empties
    assert [t.key for t in tasks] == [0, 1, 2]


# ---- the head's read list -------------------------------------------------------------------
def _uproot_source(array):
    (item,) = [
        (nid, s)
        for nid, s in array.session.sources().items()
        if isinstance(s, _GraphedTTreeSource)
    ]
    return item


def test_head_of_a_field_subset_reads_only_those_fields():
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1", "pz1"])
    sub = g[["px1", "py1"]]
    nid, source = _uproot_source(sub)
    assert _evaluation_columns(sub, nid, source._common_keys) == ("px1", "py1")

    got = ak.Array(uproot.graphed_head(sub, 3))
    raw = uproot.open(_zmumu()).arrays(["px1", "py1"], entry_stop=3)
    assert got.fields == ["px1", "py1"]
    assert ak.array_equal(got, raw)


def test_head_of_a_whole_record_operation_reads_every_selected_branch():
    g = uproot.graphed(_zmumu(), library="ak", filter_name=["px1", "py1", "pz1"])
    masked = g[g.px1 > 0]
    nid, source = _uproot_source(masked)
    assert _evaluation_columns(masked, nid, source._common_keys) == (
        "px1",
        "py1",
        "pz1",
    )

    got = ak.Array(uproot.graphed_head(masked, 4))
    raw = uproot.open(_zmumu()).arrays(["px1", "py1", "pz1"], entry_stop=4)
    assert ak.array_equal(got, raw[raw.px1 > 0])


def test_head_requires_an_uproot_backed_array():
    import graphed

    chunk = ak.Array({"x": np.arange(4.0)})
    session = graphed.Session(AwkwardBackend())
    form = AwkwardForm(ak.Array(chunk.layout.to_typetracer(forget_length=True)))
    array = session.source("plain", form=form, data=lambda: chunk)
    with pytest.raises(TypeError, match="exactly one uproot.graphed source"):
        uproot.graphed_head(array, 2)


# ---- missing-dependency hints ---------------------------------------------------------------
def test_missing_graphed_reports_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "graphed", None)
    with pytest.raises(ModuleNotFoundError, match="pip install graphed") as info:
        uproot.extras.graphed()
    assert str(info.value).startswith(
        "for uproot.graphed, install the 'graphed' package"
    )
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_missing_graphed_executors_reports_how_to_install_them(monkeypatch):
    monkeypatch.setitem(sys.modules, "graphed_executors", None)
    with pytest.raises(
        ModuleNotFoundError, match="pip install graphed-executors"
    ) as info:
        uproot.extras.graphed_executors()
    assert str(info.value).startswith(
        "for uproot.graphed_write, install the 'graphed-executors' package"
    )
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


# ---- the TTree's name when no file is opened (known_base_form=) -------------------------------
def _dir_tree(tmp_path):
    """A file whose ``TTree`` lives in a ``TDirectory``, plus the base form of that tree."""
    path = os.path.join(tmp_path, "dirtree.root")
    with uproot.recreate(path) as f:
        f.mkdir("dir")
        f["dir/Events"] = {"x": np.arange(10.0)}
    tree = uproot.open(f"{path}:dir/Events")
    return path, _get_ttree_form(ak, tree, tree.keys(), False)


@pytest.mark.parametrize("object_path", ["Events", "dir/Events", "dir/Events;1"])
def test_known_base_form_names_the_tree_from_the_object_path(tmp_path, object_path):
    # with no file opened the name comes from the object path, which may carry a TDirectory
    # prefix and a ;cycle suffix that the opened TTree's own name does not
    path, form = _dir_tree(tmp_path)
    opened = uproot.graphed(f"{path}:dir/Events", library="ak")
    blind = uproot.graphed(f"{path}:{object_path}", library="ak", known_base_form=form)
    assert blind.session.source_name(0) == opened.session.source_name(0) == "Events"


def test_known_base_form_falls_back_when_the_files_carry_no_object_path(tmp_path):
    # `files` need not name an object at all, and a source needs a name regardless
    path, form = _dir_tree(tmp_path)
    blind = uproot.graphed(path, library="ak", known_base_form=form)
    assert blind.session.source_name(0) == "events"
