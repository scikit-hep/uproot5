# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""graphed M10 remediations exercised through uproot (mvp-shortcomings findings A.2, A.3, C.9).

- buffer-level projection: a count-only analysis reports ``{branch: OFFSETS}`` (NON-empty, where
  ``necessary_columns`` reports the empty set) and ``resolve_read_branches`` serves it from the
  jagged branch's COUNTER branch — the list lengths without the payload baskets;
- compiled-IR execution: the executor path evaluates the reduced serialized IR per partition
  (no per-partition Session/re-record), bit-for-bit equal to the single pass;
- first-class blind partitions (no negative-entry_stop sentinel);
- ``graphed_write``: multi-source arrays rejected loudly; projected (field-selected) arrays write
  only their necessary branches.
"""

import os
import sys

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed_exec_local")
graphed_awkward = pytest.importorskip("graphed_awkward")

sys.path.insert(0, os.path.dirname(__file__))
import graphed_uproot_analysis as gu  # noqa: E402
from graphed import BufferNeed, evaluate_ir  # noqa: E402
from graphed_awkward import AwkwardBackend, gak  # noqa: E402
from graphed_core import Plan  # noqa: E402
from graphed_exec_local import ProcessExecutor  # noqa: E402


# ---- buffer-level projection (A.3) ---------------------------------------------------------------
def test_count_only_analysis_reports_offsets_not_the_empty_set():
    test_path = skhep_testdata.data_path("uproot-HZZ.root") + ":events"
    g = uproot.graphed(test_path, library="ak", filter_name=["Muon_Px", "Muon_Py"])
    counted = gak.num(g.Muon_Px, axis=1)
    # the column view is empty — feeding it to a reader reads NOTHING (the under-specification)
    assert uproot.necessary_columns(counted) == {"events": frozenset()}
    # the buffer view is truthful: the Muon_Px list STRUCTURE is needed
    assert uproot.necessary_buffers(counted) == {"events": {"Muon_Px": BufferNeed.OFFSETS}}


def test_offsets_need_is_served_by_the_counter_branch_without_the_payload():
    path = skhep_testdata.data_path("uproot-HZZ.root")
    tree = uproot.open(path)["events"]
    g = uproot.graphed(path + ":events", library="ak", filter_name=["Muon_Px", "Muon_Py"])
    needs = uproot.necessary_buffers(gak.num(g.Muon_Px, axis=1))["events"]

    to_read = uproot.resolve_read_branches(tree, needs)
    assert to_read == {"NMuon": "Muon_Px"}  # the counter branch, not the jagged payload

    counts = tree["NMuon"].array(library="np")
    full = tree["Muon_Px"].array(library="ak")
    assert np.array_equal(counts, ak.num(full, axis=1).to_numpy())


def test_data_needs_resolve_to_the_branches_themselves():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g = uproot.graphed(test_path, library="ak")
    needs = uproot.necessary_buffers(g.px1 + g.py1)["events"]
    assert needs == {"px1": BufferNeed.DATA, "py1": BufferNeed.DATA}
    tree = uproot.open(test_path.rsplit(":", 1)[0])["events"]
    assert uproot.resolve_read_branches(tree, needs) == {"px1": "px1", "py1": "py1"}


def test_buffer_view_collapses_to_the_column_view():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g = uproot.graphed(test_path, library="ak")
    expr = gu.analysis(g)
    buffers = uproot.necessary_buffers(expr)["events"]
    data_cols = {c for c, need in buffers.items() if need is BufferNeed.DATA}
    assert data_cols == set(uproot.necessary_columns(expr)["events"]) == set(gu.COLUMNS)


# ---- compiled-IR execution (A.2) ------------------------------------------------------------------
def test_executor_path_evaluates_the_compiled_ir_bit_for_bit():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    tasks = uproot.graphed_partitions(test_path, steps_per_file=6)
    plan = Plan(process=gu.process, combine=gu.combine, empty=gu.empty, tasks=tasks)
    result = ProcessExecutor(max_workers=3).run(plan).value
    assert np.array_equal(result, gu.single_pass(test_path))


def test_compiled_artifact_is_bytes_only_and_retargets():
    # the compiled analysis is pure bytes + source names: evaluable against ANY chunk, no Session
    compiled = gu.compiled()
    assert isinstance(compiled.ir, bytes) and compiled.source_names == ("events",)
    chunk = uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"].arrays(
        list(gu.COLUMNS), entry_start=0, entry_stop=100
    )
    (out,) = evaluate_ir(compiled, AwkwardBackend(), {"events": chunk})
    ref = chunk.px1 + chunk.py1
    assert ak.array_equal(out, ref)


def test_columns_are_wired_from_projection_not_by_hand():
    # the executor glue's read list and the recorded graph's projection cannot drift apart
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g = uproot.graphed(test_path, library="ak")
    assert set(gu.COLUMNS) == set(uproot.necessary_columns(gu.analysis(g))["events"])


# ---- first-class blind partitions (C.9) -----------------------------------------------------------
def test_blind_partitions_are_first_class_and_resolve_exactly_once():
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    tasks = uproot.graphed_partitions(test_path, steps_per_file=4, open_files=False)
    assert all(t.partition.is_blind for t in tasks)
    tree = uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"]
    resolved = [t.partition.resolve(tree.num_entries) for t in tasks]
    covered = sorted((r.entry_start, r.entry_stop) for r in resolved)
    assert covered[0][0] == 0 and covered[-1][1] == tree.num_entries
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))


def test_legacy_sentinel_partitions_still_read_correctly():
    # pre-M10 serialized plans carry the negative-entry_stop encoding; the reader honors them
    from graphed_core import Partition

    path = skhep_testdata.data_path("uproot-Zmumu.root")
    legacy = Partition(path, "events", 1, -4)  # step 1 of 4, old sentinel
    blind = Partition.blind(path, "events", 1, 4)
    a = uproot.read_graphed_partition(legacy, ["px1"])
    b = uproot.read_graphed_partition(blind, ["px1"])
    assert ak.array_equal(a.px1, b.px1)


# ---- graphed_write hardening (C.9) ----------------------------------------------------------------
def test_write_rejects_multi_source_arrays_loudly(tmp_path):
    # an array recorded over TWO uproot sources cannot be written partition-by-partition from one
    # of them; the guard must say so instead of silently picking the first source
    import graphed
    from graphed_awkward import AwkwardBackend, AwkwardForm
    from uproot._graphed import _GraphedTTreeSource

    path = skhep_testdata.data_path("uproot-Zmumu.root")
    chunk = uproot.open(path)["events"].arrays(["px1"], entry_stop=4)
    s = graphed.Session(AwkwardBackend())
    arrays = []
    for name in ("a", "b"):
        src = _GraphedTTreeSource([(path, "events")], ["px1"], None, False, {"num_workers": 1})
        form = AwkwardForm(ak.Array(chunk.layout.to_typetracer(forget_length=True)))
        arrays.append(s.source(name, form=form, data=src))
    with pytest.raises(TypeError, match="exactly one uproot.graphed source"):
        uproot.graphed_write(arrays[0].px1 + arrays[1].px1, os.path.join(tmp_path, "out"))


def test_write_of_a_projected_array_writes_only_its_branches(tmp_path):
    src_path = os.path.join(tmp_path, "in.root")
    with uproot.recreate(src_path) as f:
        f["events"] = {
            "x": np.arange(12, dtype="f8"),
            "y": np.arange(12, dtype="f8") * 2.0,
            "z": np.arange(12, dtype="f8") * 3.0,
        }
    g = uproot.graphed(src_path + ":events", library="ak")
    outdir = os.path.join(tmp_path, "out")
    paths = uproot.graphed_write(g.x + g.y, outdir, steps_per_file=2, tree_name="events")
    back = uproot.open(paths[0] + ":events").arrays()
    assert set(back.fields) == {"x", "y"}, "the unused branch z must not be written"
