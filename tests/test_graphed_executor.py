# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Run a uproot-read ``graphed`` analysis through the ``graphed-exec-local`` executors.

The recorded graph is executed per ``Partition`` and tree-reduced across worker processes — the real
process/thread-executor path that a deferred-array ``.compute()`` hides — and the result must match
the single-pass plain-uproot computation bit-for-bit, regardless of how the events are chunked.
"""

import os
import sys

import numpy as np
import pytest
import skhep_testdata

import uproot  # noqa: F401  (registers uproot.graphed_partitions used by the helper)

pytest.importorskip("graphed_exec_local")
pytest.importorskip("graphed_awkward")

# make the picklable helper importable both here and in spawned ProcessExecutor workers
# (multiprocessing 'spawn' inherits this sys.path)
sys.path.insert(0, os.path.dirname(__file__))
import graphed_uproot_analysis as gu
from graphed_core import Plan
from graphed_exec_local import ProcessExecutor, ThreadExecutor

ZMUMU = "uproot-Zmumu.root:events"


def _plan(path_with_tree: str, n_chunks: int) -> Plan:
    return Plan(
        process=gu.process,
        combine=gu.combine,
        empty=gu.empty,
        tasks=gu.partitions(path_with_tree, n_chunks),
    )


@pytest.mark.parametrize("Executor", [ThreadExecutor, ProcessExecutor])
def test_uproot_graphed_via_executor_matches_single_pass(Executor):
    path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    full = gu.single_pass(path)

    result = Executor(max_workers=4).run(_plan(path, n_chunks=6))

    assert np.array_equal(result.value, full), f"{Executor.__name__}: chunked != single pass"
    assert result.n_combines == 5  # 6 chunks -> 5 tree-reduce combines


def test_process_executor_result_is_invariant_to_partition_count():
    path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    full = gu.single_pass(path)
    for n_chunks in (1, 4, 13):
        result = ProcessExecutor(max_workers=4).run(_plan(path, n_chunks))
        assert np.array_equal(result.value, full), f"{n_chunks} chunks changed the histogram"


def test_executor_histogram_is_non_vacuous():
    # guard against a trivially-passing test: the analysis must actually fill the histogram
    path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    assert int(gu.single_pass(path).sum()) > 0
