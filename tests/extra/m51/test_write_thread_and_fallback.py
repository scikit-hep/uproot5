# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Extra (non-frozen) m51 coverage: drive ``_write_partition`` in-process via the THREAD executor
so plain coverage sees the worker body (the frozen suite runs it in spawn ProcessPool children,
invisible to a non-subprocess coverage run), and exercise the bare-expression fallback branch the
frozen suite never reaches (frozen writes only records).

Both are real behavioral assertions, not coverage no-ops."""
import os

import awkward as ak
import numpy as np
import pytest

import uproot

pytest.importorskip("graphed_executors.local")
pytest.importorskip("graphed.awkward")

from graphed.awkward import gak  # noqa: E402


def _src(tmp_path, n=12):
    p = os.path.join(tmp_path, "in.root")
    with uproot.recreate(p) as f:
        f["events"] = {"x": np.arange(n, dtype="f8"), "y": np.arange(n, dtype="f8") * 2.0}
    return p + ":events"


def test_derived_record_thread_executor(tmp_path):
    """Record path (``evaluated.fields`` truthy) via ``executor="thread"`` — same derived-column
    write as the frozen suite, but in-process so the worker body is covered without subprocess
    coverage."""
    g = uproot.graphed(_src(tmp_path), library="ak")
    rec = gak.zip({"x": g.x, "doubled": g.x * 2.0 + 1.0})
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(rec, outdir, steps_per_file=3, tree_name="events", executor="thread")

    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    ref = uproot.open(_src(tmp_path)).arrays(["x"])
    assert set(back.fields) == {"x", "doubled"}
    assert ak.array_equal(back["doubled"], ref.x * 2.0 + 1.0)


def test_bare_expression_falls_back_to_source_columns(tmp_path):
    """Fallback branch (``evaluated.fields`` empty → source columns): a bare non-record expression
    ``g.x + g.y`` has no field to name, so the write falls back to the projected source columns
    (pre-m51 behavior). Thread executor keeps it in-process for coverage."""
    src = _src(tmp_path)
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(g.x + g.y, outdir, steps_per_file=2, tree_name="events", executor="thread")

    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    ref = uproot.open(src).arrays()
    assert set(back.fields) == {"x", "y"}  # source columns, not the (nameless) sum
    assert ak.array_equal(back["x"], ref.x)
    assert ak.array_equal(back["y"], ref.y)


def test_syntactic_read_list_witness_no_starve(tmp_path):
    """Witness that the read list is the SYNTACTIC ``_evaluation_columns``, not the finer
    ``necessary_columns`` buffer projection (§6.4f). ``zip({"a": g.x, "b": g.y})[["a"]]`` keeps only
    field ``a`` (=x) in the OUTPUT, so ``necessary_columns`` = {x} — but the ``zip`` node
    syntactically REPLAYS a read of ``y``, so evaluation needs {x, y}. With ``necessary_columns`` the
    worker would read only x and ``evaluate_ir`` would STARVE (the awkward backend raises
    ``no field named 'y'``); ``_evaluation_columns`` = {x, y} feeds it. This PASSES today and would
    fail if ``graphed_write`` used ``necessary_columns`` (demonstrated in the m51 attempts log)."""
    src = _src(tmp_path)
    g = uproot.graphed(src, library="ak")
    rec = gak.zip({"a": g.x, "b": g.y})[["a"]]
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(rec, outdir, steps_per_file=2, tree_name="events", executor="thread")

    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    ref = uproot.open(src).arrays(["x"])
    assert set(back.fields) == {"a"}
    assert ak.array_equal(back["a"], ref.x)
