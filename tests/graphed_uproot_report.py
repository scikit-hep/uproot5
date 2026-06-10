# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Glue for report-style (dead-letter) testing of a uproot-read ``graphed`` analysis through
``graphed-checkpoint``'s ``run_resumable``.

This is graphed's own error-harvesting model, not a copy of dask's report tuple: a ``DurablePlan``
runs each ``Partition`` through ``process`` (read the chunk via uproot, histogram it); a missing/broken
file raises, and ``run_resumable`` harvests that partition into the content-addressed **dead-letter
set** (``ResumeReport.dead`` / ``dead_letters``) with a reproducible descriptor, while the good
partitions still reduce to a result. The same store gives resume (skip completed, no double-count) and
an error budget as a stopping condition.

Every callable is module-level so a ``DurablePlan`` can reference it by import path (``OpSpec.from_ref``).
"""
from __future__ import annotations

import awkward as ak
import numpy as np
import uproot
from graphed import Session
from graphed_awkward import AwkwardBackend, from_awkward
from graphed_core import DurablePlan, GraphStore, OpSpec

COLUMN = "px1"
BINS, LO, HI = 50, -150.0, 150.0


def _counts(values) -> np.ndarray:
    return np.histogram(ak.to_numpy(values), bins=BINS, range=(LO, HI))[0].astype(np.int64)


def process(partition, resources):
    # read this partition via uproot (a missing/broken file raises -> run_resumable dead-letters it),
    # then record + materialize the analysis on the chunk
    arr = uproot.read_graphed_partition(partition, [COLUMN])
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", arr)
    return _counts(s.materialize(ev.px1))


def hist_add(a, b):
    return a + b


def hist_zero():
    return np.zeros(BINS, dtype=np.int64)


def _ir() -> bytes:
    # a small representative IR so the content-addressed task_id is meaningful + deterministic
    g = GraphStore()
    src = g.add_source("events", {"uri": "uproot://events"})
    px = g.add_op("field", [src], {"field": COLUMN})
    out = g.add_reduction("hist", [px], {"bins": BINS})
    return g.serialize(outputs=[out])  # [freeze-M22-1: mark_output removed; outputs per request]


def build_plan(partitions, *, error_budget=None) -> DurablePlan:
    return DurablePlan(
        ir=_ir(),
        process=OpSpec.from_ref("graphed_uproot_report:process"),
        combine=OpSpec.from_ref("graphed_uproot_report:hist_add"),
        empty=OpSpec.from_ref("graphed_uproot_report:hist_zero"),
        partitions=tuple(partitions),
        read_columns=(COLUMN,),
        stopping={} if error_budget is None else {"error_budget": int(error_budget)},
    )


def single_pass(path_with_tree: str) -> np.ndarray:
    arr = uproot.open(path_with_tree).arrays([COLUMN], library="ak")
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", arr)
    return _counts(s.materialize(ev.px1))
