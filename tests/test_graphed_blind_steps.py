# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Blind steps for ``uproot.graphed_partitions`` — the analogue of ``test_0876`` for graphed.

``step_size`` / ``steps_per_file`` / ``open_files`` follow the same rules as ``uproot.dask``:
``step_size`` is incompatible with ``open_files=False`` and mutually exclusive with ``steps_per_file``.
With ``open_files=False`` (blind), files are not opened to build the partitions; each chunk's real
entry range is resolved against the file's own count when it is read. However the dataset is chunked,
the executor's tree-reduced result must equal the single-pass computation.
"""

import os
import sys

import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed_exec_local")
pytest.importorskip("graphed_awkward")

sys.path.insert(0, os.path.dirname(__file__))
import graphed_uproot_analysis as gu
from graphed_core import Plan
from graphed_exec_local import ProcessExecutor


def _run(tasks):
    plan = Plan(process=gu.process, combine=gu.combine, empty=gu.empty, tasks=tasks)
    return ProcessExecutor(max_workers=4).run(plan).value


@pytest.mark.parametrize("step_size", ["50 kB", uproot._util.unset])
@pytest.mark.parametrize("steps_per_file", [1, 2, 5, 13, uproot._util.unset])
@pytest.mark.parametrize("open_files", [False, True])
def test_graphed_blind_steps(step_size, steps_per_file, open_files):
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    have_step_size = not isinstance(step_size, uproot._util._Unset)
    have_steps_per_file = not isinstance(steps_per_file, uproot._util._Unset)

    if have_step_size and not open_files:
        with pytest.raises(TypeError):
            uproot.graphed_partitions(
                test_path, step_size=step_size, steps_per_file=steps_per_file, open_files=open_files
            )
    elif have_step_size and have_steps_per_file:
        with pytest.raises(TypeError):
            uproot.graphed_partitions(
                test_path, step_size=step_size, steps_per_file=steps_per_file, open_files=open_files
            )
    else:
        tasks = uproot.graphed_partitions(
            test_path, step_size=step_size, steps_per_file=steps_per_file, open_files=open_files
        )
        # however the events are chunked (blind or not), the result matches the single-pass histogram
        assert np.array_equal(_run(tasks), gu.single_pass(test_path))


def test_blind_partitions_do_not_open_the_file():
    # open_files=False must build partitions without reading entry counts: blind chunks are
    # FIRST-CLASS graphed_core blind partitions carrying (step, n_steps) explicitly, resolved only
    # at read time. (freeze-UPROOT-1 amendment: this previously pinned the negative-entry_stop
    # sentinel encoding, which graphed-core M10 retired — same intent, honest representation.)
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    tasks = uproot.graphed_partitions(test_path, steps_per_file=5, open_files=False)
    assert len(tasks) == 5
    assert all(t.partition.is_blind for t in tasks)
    assert sorted(t.partition.blind_step for t in tasks) == [0, 1, 2, 3, 4]
    assert all(t.partition.blind_n_steps == 5 for t in tasks)
    # no sentinel left behind: the entry range is genuinely unset until resolve()
    assert all(t.partition.entry_start == 0 and t.partition.entry_stop == 0 for t in tasks)


def test_blind_and_eager_partitions_agree():
    # blind (open_files=False) and eager (open_files=True) chunking give the same executor result
    test_path = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    blind = _run(uproot.graphed_partitions(test_path, steps_per_file=7, open_files=False))
    eager = _run(uproot.graphed_partitions(test_path, steps_per_file=7, open_files=True))
    assert np.array_equal(blind, eager)
    assert np.array_equal(blind, gu.single_pass(test_path))
