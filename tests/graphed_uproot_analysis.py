# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Picklable glue to run a uproot-read ``graphed`` analysis through the ``graphed-executors``
executors (the M7 ``adl.py`` pattern, but reading from a real ROOT file).

A ``Partition`` is a ``(file, tree, entry_start, entry_stop)`` chunk. ``process`` opens the file
**once per worker** (``open_once``), reads only the ``TBranches`` the analysis needs for that entry
range, and evaluates the COMPILED, REDUCED IR on the chunk (graphed M10 ``compile_ir`` /
``evaluate_ir``: the analysis is recorded and optimized ONCE per worker — cached at module level —
then each partition costs one backend dispatch per reduced node, with no per-partition Session and
no re-recording; the dask #2/#6/#7 failure mode this project exists to avoid). The executor
tree-reduces the per-chunk histograms across worker processes into one histogram that must match
the single-pass plain-uproot result bit-for-bit.

This is the real thread/process-executor path that a deferred-array ``.compute()`` hides — so the
uproot ``graphed`` tests exercise ``ProcessPoolExecutor`` instead of a thin ``materialize`` wrapper. Every
callable here is module-level so a spawned ``ProcessPoolExecutor`` worker can import it by reference.
"""
from __future__ import annotations

import awkward as ak
import numpy as np
import uproot
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.core import Partition, Task

# The recorded analysis touches exactly these TBranches; the executor reads only them.
COLUMNS = ("px1", "py1")
BINS, LO, HI = 50, -200.0, 200.0


def analysis(events):
    """The recorded graph evaluated per chunk: a deferred ``graphed`` expression (``px1 + py1``)."""
    return events.px1 + events.py1


def _counts(values) -> np.ndarray:
    return np.histogram(ak.to_numpy(values), bins=BINS, range=(LO, HI))[0].astype(np.int64)


_COMPILED = None


def compiled():
    """The analysis compiled to its reduced serialized IR — once per worker, cached. Recording
    needs only the source FORM (a one-row stand-in with the branches' dtypes), never event data."""
    global _COMPILED
    if _COMPILED is None:
        from graphed import compile_ir

        s = Session(AwkwardBackend(), incremental=True)
        ev = from_awkward(s, "events", ak.Array({c: [0.0] for c in COLUMNS}))
        _COMPILED = compile_ir(s, analysis(ev))
    return _COMPILED


def process(part: Partition, res: object) -> np.ndarray:
    # open the file once per worker (HDD locality), read only this chunk's needed columns; the reader
    # resolves blind partitions (open_files=False) against the file's actual entry count
    from graphed import evaluate_ir

    tree = res.open_once(part.uri, uproot.open)[part.tree]  # type: ignore[attr-defined]
    chunk = uproot.read_graphed_partition(part, COLUMNS, tree=tree)
    (out,) = evaluate_ir(compiled(), AwkwardBackend(), {"events": chunk})
    return _counts(out)  # evaluate the REDUCED IR on the chunk — no Session, no re-record


def combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def empty() -> np.ndarray:
    return np.zeros(BINS, dtype=np.int64)


def partitions(path_with_tree: str, n_chunks: int) -> list[Task]:
    return uproot.graphed_partitions(path_with_tree, steps_per_file=n_chunks)


def single_pass(path_with_tree: str) -> np.ndarray:
    """The bit-for-bit target: read the whole tree, record + materialize once, then histogram."""
    arr = uproot.open(path_with_tree).arrays(list(COLUMNS), library="ak")
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", arr)
    return _counts(s.materialize(analysis(ev)))
