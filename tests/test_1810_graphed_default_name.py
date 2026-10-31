# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Default-named ``uproot.graphed`` reads of different data share a ``Session``."""

from __future__ import annotations

import os

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(graphed, "current_session"), reason="graphed has no current Session"
)


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _names(session):
    return [session.source_name(nid) for nid in session.source_ids()]


def _events(tmp_path, stem, offset=0.0):
    path = os.path.join(tmp_path, f"{stem}.root")
    with uproot.recreate(path) as f:
        f.mktree("Events", {"x": "f8", "y": "f8"})
        f["Events"].extend({"x": np.arange(3.0) + offset, "y": np.arange(3.0)})
    return path + ":Events"


def _ir(array):
    return bytes(graphed.compile_ir(array.session, array).ir)


def test_a_default_read_of_other_files_gets_its_own_source(tmp_path):
    dy = uproot.graphed(_events(tmp_path, "dy"))
    tt = uproot.graphed(_events(tmp_path, "tt", 100.0))
    assert dy.session is tt.session
    first, second = _names(dy.session)
    assert first == "Events"
    assert second.startswith("Events-")
    assert uproot._graphed.necessary_columns(tt.x) == {second: frozenset({"x"})}
    assert graphed.materialize(tt.x).tolist() == [100.0, 101.0, 102.0]
    assert graphed.materialize(dy.y).tolist() == [0.0, 1.0, 2.0]


def test_a_default_read_of_another_selection_gets_its_own_source(tmp_path):
    path = _events(tmp_path, "dy")
    whole = uproot.graphed(path)
    only_x = uproot.graphed(path, filter_name="x")
    assert only_x.node_id != whole.node_id
    assert only_x.fields == ["x"]
    assert graphed.materialize(only_x.x).tolist() == [0.0, 1.0, 2.0]


def test_every_default_read_interns_on_a_reread(tmp_path):
    dy = _events(tmp_path, "dy")
    reads = [(dy, {}), (_events(tmp_path, "tt"), {}), (dy, {"filter_name": "x"})]
    arrays = [uproot.graphed(path, **options) for path, options in reads]
    session = arrays[0].session
    count = session.node_count()
    again = [uproot.graphed(path, **options) for path, options in reads]
    assert [a.node_id for a in again] == [a.node_id for a in arrays]
    assert session.node_count() == count == 3


def test_a_second_spelling_of_a_tree_interns(tmp_path):
    path = _events(tmp_path, "dy")
    other = uproot.graphed(_events(tmp_path, "tt"))
    first = uproot.graphed(path)
    assert first.node_id != other.node_id
    assert uproot.graphed(path + ";1").node_id == first.node_id


def test_a_read_between_two_others_shifts_no_name(tmp_path):
    dy, tt, ww = (_events(tmp_path, stem) for stem in ("dy", "tt", "ww"))
    uproot.graphed(dy)
    uproot.graphed(tt)
    three = _names(uproot.graphed(ww).session)
    graphed.new_session(None)
    uproot.graphed(dy)
    two = _names(uproot.graphed(ww).session)
    assert two == [three[0], three[2]]


def test_a_first_read_keeps_the_ir_of_a_read_alone(tmp_path):
    dy = _events(tmp_path, "dy")
    alone = _ir(uproot.graphed(dy).x)
    graphed.new_session(None)
    first = uproot.graphed(dy)
    uproot.graphed(_events(tmp_path, "tt"))
    assert _ir(first.x) == alone


def test_reads_the_digest_cannot_tell_apart_are_numbered():
    executors = [uproot.source.futures.TrivialExecutor() for _ in range(3)]
    arrays = [
        uproot.graphed(_zmumu(), decompression_executor=executor)
        for executor in executors
    ]
    first, hashed, numbered = _names(arrays[0].session)
    assert first == "events"
    assert hashed.startswith("events-")
    assert numbered == f"{hashed}#1"
    assert (
        uproot.graphed(_zmumu(), decompression_executor=executors[2]).node_id
        == arrays[2].node_id
    )


def test_a_refusal_of_a_free_name_is_not_retried(monkeypatch):
    def refuse(self, name, *, form, data):
        raise ValueError(f"no {name}")

    monkeypatch.setattr(graphed.Session, "source", refuse)
    with pytest.raises(ValueError, match="no events"):
        uproot.graphed(_zmumu())


def test_equal_trees_in_two_directories_are_told_apart(tmp_path):
    path = os.path.join(tmp_path, "dirs.root")
    with uproot.recreate(path) as f:
        for directory in "abc":
            f.mktree(f"{directory}/Events", {"x": "f8"}).extend({"x": np.arange(3.0)})
    reads = [uproot.graphed(f"{path}:{directory}/Events") for directory in "abc"]
    names = _names(reads[0].session)
    assert len(names) == 3
    assert not any("#" in name for name in names)
    assert uproot.graphed(f"{path}:/c/Events;1").node_id == reads[2].node_id


def test_each_digest_part_tells_reads_apart(tmp_path):
    path = _events(tmp_path, "dy")
    steps = {path.rpartition(":")[0]: {"object_path": "Events", "steps": [[0, 1]]}}
    uproot.graphed(path)
    # the digest cannot see these, so this read takes the digest of the one above
    session = uproot.graphed(path, custom_classes=dict(uproot.classes)).session
    uproot.graphed(path, allow_missing=True)
    uproot.graphed(path, align_baskets=True)
    uproot.graphed(path, filter_name="x")
    uproot.graphed(steps)
    uproot.graphed(_events(tmp_path, "tt"))
    names = _names(session)
    assert len(names) == 7
    assert not any("#" in name for name in names)
