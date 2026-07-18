# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Report-style (dead-letter) testing of uproot-read graphed analyses via ``graphed-checkpoint``.

graphed semantics (not dask's report tuple): ``run_resumable`` harvests a failed partition into the
content-addressed dead-letter set while the good partitions still reduce; the same store gives resume
(skip completed, no double-count) and an error budget as a stopping condition.
"""

import os
import sys

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.checkpoint")
pytest.importorskip("graphed.awkward")

sys.path.insert(0, os.path.dirname(__file__))
import graphed_uproot_report as gr
from graphed.checkpoint import Store, run_resumable
from graphed.checkpoint.runner import _SimulatedInterrupt
from graphed.core import Partition


def _whole_file(path_with_tree):
    path, tree = path_with_tree.rsplit(":", 1)
    n = uproot.open(path_with_tree).num_entries
    return Partition(path, tree, 0, n)


def _missing(tmp_path, name="missing.root"):
    return Partition(str(tmp_path / name), "events", 0, 1)


def test_bad_file_is_dead_lettered_and_good_partitions_reduce(tmp_path):
    p1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    p2 = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"
    plan = gr.build_plan([_whole_file(p1), _whole_file(p2), _missing(tmp_path)])

    res = run_resumable(plan, Store(str(tmp_path / "store")))

    # the missing file is harvested into the dead-letter set; the two good partitions still reduce
    assert res.report.executed == 2
    assert res.report.dead == 1
    (dl,) = res.report.dead_letters
    assert dl["task_id"] == plan.task_id(_missing(tmp_path))  # reproducible content-addressed id
    assert np.array_equal(res.value, gr.single_pass(p1) + gr.single_pass(p2))


def test_clean_run_has_no_dead_letters(tmp_path):
    p1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    plan = gr.build_plan([_whole_file(p1)])

    res = run_resumable(plan, Store(str(tmp_path)))

    assert res.report.dead == 0
    assert res.report.dead_letters == []
    assert np.array_equal(res.value, gr.single_pass(p1))


def test_kill_then_resume_skips_completed_and_matches(tmp_path):
    p1 = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    parts = [t.partition for t in uproot.graphed_partitions(p1, steps_per_file=4)]
    plan = gr.build_plan(parts)
    store = Store(str(tmp_path))

    # crash after 2 partitions commit, then resume on the same store
    with pytest.raises(_SimulatedInterrupt):
        run_resumable(plan, store, _kill_after=2)
    assert len(store.completed()) == 2

    res = run_resumable(plan, store)
    assert res.report.skipped == 2  # completed work reused, not recomputed
    assert res.report.executed == 2  # only the unfinished partitions run
    assert res.report.did_less_work
    assert np.array_equal(res.value, gr.single_pass(p1))  # resumed result == single pass


def test_error_budget_stops_the_run(tmp_path):
    bads = [_missing(tmp_path, f"missing{i}.root") for i in range(3)]
    plan = gr.build_plan(bads, error_budget=0)

    res = run_resumable(plan, Store(str(tmp_path / "store")))

    assert res.report.stopped == "error_budget"
    assert res.report.dead == 1  # stops once the dead count (1) exceeds the budget (0)
