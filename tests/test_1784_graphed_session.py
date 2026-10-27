# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed(session=, name=)``: a read recorded into an existing ``Session``."""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")

import graphed  # noqa: E402
from graphed.awkward import AwkwardBackend  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed_helpers import mini_nanoaod as mini  # noqa: E402


def _zmumu():
    return skhep_testdata.data_path("uproot-Zmumu.root") + ":events"


def _nano():
    return skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"


def _keyed_tree(path, field, values):
    with uproot.recreate(path) as f:
        f["events"] = {
            "k": np.arange(len(values), dtype="i8"),
            field: np.asarray(values, "f8"),
        }
    return path + ":events"


def _joined(tmp_path):
    left = uproot.graphed(
        _keyed_tree(os.path.join(tmp_path, "a.root"), "x", [1.0, 2.0, 3.0])
    )
    right = uproot.graphed(
        _keyed_tree(os.path.join(tmp_path, "b.root"), "z", [10.0, 20.0]),
        session=left.session,
        name="friends",
    )
    return left, right, graphed.join(left, right, on=["k"])


def test_session_and_backend_are_exclusive():
    with pytest.raises(TypeError, match="pass backend= or session=, not both"):
        uproot.graphed(
            _zmumu(), backend=AwkwardBackend(), session=uproot.graphed(_zmumu()).session
        )
    with pytest.raises(TypeError, match="behavior= is for the default backend"):
        uproot.graphed(_zmumu(), behavior={}, session=uproot.graphed(_zmumu()).session)


def test_a_read_into_another_reads_session_joins_with_it(tmp_path):
    left, right, joined = _joined(tmp_path)
    session = left.session
    assert right.session is session
    assert [session.source_name(nid) for nid in session.source_ids()] == [
        "events",
        "friends",
    ]
    assert session.form(right).describe() == "## * {k: int64, z: float64}"

    plan = graphed.join_plan(
        joined.x + joined.z,
        reduce=lambda vals: sorted(ak.to_list(vals[0])),
        combine=lambda a, b: sorted(a + b),
        empty=list,
    )
    assert SequentialRunner().run(plan).value == [11.0, 22.0]
    assert uproot.graphed_head(right.z, 5).tolist() == [10.0, 20.0]


def test_session_must_be_an_awkward_session():
    with pytest.raises(TypeError, match="session= takes a graphed.Session, not "):
        uproot.graphed(_zmumu(), session=uproot.graphed(_zmumu()))
    from graphed.numpy import NumpyBackend

    with pytest.raises(
        TypeError, match="records on NumpyBackend, and this read on AwkwardBackend"
    ):
        uproot.graphed(_zmumu(), session=graphed.Session(NumpyBackend()))


def test_form_mapping_needs_the_sessions_behavior():
    def read(session, name, **kwargs):
        return uproot.graphed(
            _nano(),
            filter_name=mini.FILTER,
            form_mapping=mini.MiniNanoMapping(**kwargs),
            session=session,
            name=name,
        )

    with pytest.raises(
        TypeError,
        match="records on AwkwardBackend with another behavior, and this read on "
        "AwkwardBackend with its form_mapping behavior",
    ):
        read(uproot.graphed(_zmumu()).session, "nano")
    mapped = uproot.graphed(
        _nano(), filter_name=mini.FILTER, form_mapping=mini.MiniNanoMapping()
    )
    assert read(mapped.session, "nano").session is mapped.session
    with pytest.raises(TypeError, match="with another behavior"):
        read(mapped.session, "other", behavior={})


def test_a_taken_name_is_refused():
    path = _zmumu()
    session = uproot.graphed(path).session
    bare = path.rpartition(":")[0]  # the name comes from the opened TTree
    for files, name in ((path, None), (path, "events"), (bare, None)):
        with pytest.raises(ValueError, match="'events'; pass name= to name this read"):
            uproot.graphed(files, session=session, name=name)
    mine = uproot.graphed(path, name="mine")
    assert [mine.session.source_name(nid) for nid in mine.session.source_ids()] == [
        "mine"
    ]


def test_a_known_taken_name_is_refused_before_any_file_opens(tmp_path):
    session = uproot.graphed(_zmumu()).session
    missing = os.path.join(tmp_path, "missing.root")
    for files, name in ((missing + ":events", None), (missing, "events")):
        with pytest.raises(ValueError, match="'events'; pass name= to name this read"):
            uproot.graphed(files, session=session, name=name)


def test_a_branch_path_is_not_refused_under_the_branch_name():
    session = uproot.graphed(_zmumu(), name="px1").session
    leaf = uproot.graphed(_zmumu() + "/px1", session=session)
    assert [session.source_name(nid) for nid in session.source_ids()] == [
        "px1",
        "events",
    ]
    assert leaf.session is session


def test_head_finds_its_source_by_node_not_name(tmp_path):
    left, _, _ = _joined(tmp_path)
    session = left.session
    # another integration's source under the same name, kept apart by its params
    other = session.source("events", form=session.form(left), data=None, tag=1)
    assert len(session.source_ids()) == 3
    with pytest.raises(TypeError, match="this array is backed by 0"):
        uproot.graphed_head(other.x)


def test_necessary_columns_projects_once_over_the_arrays_own_sources(
    tmp_path, monkeypatch
):
    left, _, joined = _joined(tmp_path)
    assert uproot._graphed.necessary_columns(left.x) == {"events": frozenset({"x"})}

    calls = []
    project = graphed.awkward.projection.project_buffers_many

    def counting(*args, **kwargs):
        calls.append(args)
        return project(*args, **kwargs)

    monkeypatch.setattr(graphed.awkward.projection, "project_buffers_many", counting)
    assert uproot._graphed.necessary_columns(joined.x + joined.z) == {
        "events": frozenset({"k", "x"}),
        "friends": frozenset({"k", "z"}),
    }
    assert len(calls) == 1


def test_head_and_write_count_sources_before_compiling(tmp_path, monkeypatch):
    _, _, joined = _joined(tmp_path)

    def no_compile(*args, **kwargs):
        raise AssertionError("compiled before counting sources")

    monkeypatch.setattr(graphed, "compile_ir", no_compile)
    with pytest.raises(TypeError, match="graphed_head .* backed by 2"):
        uproot.graphed_head(joined.x + joined.z)
    with pytest.raises(TypeError, match="graphed_write .* backed by 2"):
        uproot.graphed_write(joined, os.path.join(tmp_path, "out"))
