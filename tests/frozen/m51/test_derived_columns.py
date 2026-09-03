# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Frozen acceptance — m51 anchor K: ROOT DERIVED-COLUMN round-trip through ``graphed_write``.

``uproot.graphed_write`` today copies source ``TBranch``\\ es VERBATIM: ``_write_partition`` builds
``record = {name: chunk[name] for name in chunk.fields}`` from the RAW partition read, with no
``compile_ir``/``evaluate_ir``. A graphed graph carrying a DERIVED column — a field computed from
an expression, absent from the source branches — is therefore lost on write: the output tree
carries the raw branches the graph reads, never the computed field. m51/R1 makes ``graphed_write``
compile the recorded array and evaluate it PER PARTITION, so the derived column is written and
round-trips through a plain ``uproot.open``.

Anchor K (``vary-m51-decomposition.md`` §3, commit R1). NON-VARIED only — varied ROOT write-out is
Phase-2 (plan §11); nothing here reads a manifest or a per-universe reconstruction.

Discriminating (proven against the verbatim-copy write at authoring time): the derived-column
assertions FAIL today because the computed field is absent from the output; the plain-source test
is the POSITIVE CONTROL that DOES pass today, isolating the failures to the missing IR evaluation
rather than a broken write/read harness.
"""
import os

import awkward as ak
import numpy as np
import pytest

import uproot

pytest.importorskip("graphed_executors.local")
pytest.importorskip("graphed.awkward")

from graphed.awkward import gak  # noqa: E402


def _make_flat_root(path, n=20):
    with uproot.recreate(path) as f:
        f["events"] = {"x": np.arange(n, dtype="f8"), "y": np.arange(n, dtype="f8") * 3.0}
    return path + ":events"


def _make_jagged_root(path):
    # a realistic skim shape: one jagged per-object branch with variable multiplicity (incl. empty)
    jet = ak.Array([[10.0, 20.0], [30.0], [], [40.0, 50.0, 60.0], [70.0]])
    with uproot.recreate(path) as f:
        f["Events"] = {"Jet_pt": jet}
    return path + ":Events", jet


def test_plain_source_roundtrips_positive_control(tmp_path):
    """POSITIVE CONTROL: a plain (non-derived) graphed read round-trips through ``graphed_write``
    TODAY — the raw branches are copied verbatim across every partition. This proves the
    write/read harness itself works, so the derived-column failures below are the missing IR
    evaluation, not a broken harness."""
    src = _make_flat_root(os.path.join(tmp_path, "in.root"), n=20)
    g = uproot.graphed(src, library="ak")
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(g, outdir, steps_per_file=4, tree_name="events")

    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    ref = uproot.open(src).arrays()
    assert set(back.fields) == {"x", "y"}
    assert ak.array_equal(back.x, ref.x)
    assert ak.array_equal(back.y, ref.y)


def test_derived_column_roundtrips(tmp_path):
    """A record pairing a passthrough field (``x``) with a DERIVED field (``doubled = x*2 + 1``)
    computed from a source branch. Written over MULTIPLE partitions, so the per-partition IR
    evaluation (not a single whole-array eval) is exercised: each part must carry the computed
    values for its own entry range. Today's verbatim copy writes only the raw branch it reads, so
    ``doubled`` is absent from the output."""
    src = _make_flat_root(os.path.join(tmp_path, "in.root"), n=20)
    g = uproot.graphed(src, library="ak")
    rec = gak.zip({"x": g.x, "doubled": g.x * 2.0 + 1.0})
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(rec, outdir, steps_per_file=4, tree_name="events")

    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    ref = uproot.open(src).arrays(["x"])
    assert "doubled" in back.fields, f"derived column dropped; wrote {sorted(back.fields)}"
    assert ak.array_equal(back["doubled"], ref.x * 2.0 + 1.0)
    # the passthrough field is the in-test control: a plain-value field written beside the derived
    assert ak.array_equal(back["x"], ref.x)


def test_pure_derived_record_is_named_by_its_field(tmp_path):
    """A record whose ONLY field is derived (``energy = sqrt(x^2 + y^2)``). The output tree must
    carry the DERIVED field name — not the raw source branches the expression reads. Today's
    verbatim copy names the output after the raw branches (``x``, ``y``), never ``energy``."""
    src = _make_flat_root(os.path.join(tmp_path, "in.root"), n=12)
    g = uproot.graphed(src, library="ak")
    rec = gak.zip({"energy": np.sqrt(g.x * g.x + g.y * g.y)})
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(rec, outdir, steps_per_file=3, tree_name="events")

    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    ref = uproot.open(src).arrays(["x", "y"])
    assert set(back.fields) == {"energy"}
    assert ak.array_equal(back["energy"], np.sqrt(ref.x * ref.x + ref.y * ref.y))


def test_jagged_derived_column_roundtrips(tmp_path):
    """A derived per-object kinematic (``Jet_pt2 = Jet_pt * 2``) over a jagged branch: the derived
    column must be computed AND its list structure preserved. Verbatim copy drops it."""
    src, jet = _make_jagged_root(os.path.join(tmp_path, "in.root"))
    g = uproot.graphed(src, library="ak")
    rec = gak.zip({"Jet_pt": g.Jet_pt, "Jet_pt2": g.Jet_pt * 2.0})
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(rec, outdir, steps_per_file=1, tree_name="Events")

    back = ak.concatenate([uproot.open(p + ":Events").arrays() for p in paths])
    assert "Jet_pt2" in back.fields, f"derived jagged column dropped; wrote {sorted(back.fields)}"
    assert ak.array_equal(back["Jet_pt2"], jet * 2.0)
    assert ak.array_equal(back["Jet_pt"], jet)
