# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed_write`` writes DERIVED columns.

The write compiles the recorded graph and evaluates it PER PARTITION, so a field computed from an
expression -- absent from the source ``TBranch``\\ es -- is written and round-trips through a plain
``uproot.open``. Non-varied only.
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
        f["events"] = {
            "x": np.arange(n, dtype="f8"),
            "y": np.arange(n, dtype="f8") * 3.0,
        }
    return path + ":events"


def _make_jagged_root(path):
    # a realistic skim shape: one jagged per-object branch with variable multiplicity (incl. empty)
    jet = ak.Array([[10.0, 20.0], [30.0], [], [40.0, 50.0, 60.0], [70.0]])
    with uproot.recreate(path) as f:
        f["Events"] = {"Jet_pt": jet}
    return path + ":Events", jet


def test_plain_source_roundtrips_positive_control(tmp_path):
    """Control: a plain (non-derived) graphed read round-trips, the raw branches copied verbatim
    across every partition."""
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
    """A record pairing a passthrough field (``x``) with a derived field (``doubled = x*2 + 1``),
    written over MULTIPLE partitions so per-partition evaluation is exercised: each part carries
    the computed values for its own entry range."""
    src = _make_flat_root(os.path.join(tmp_path, "in.root"), n=20)
    g = uproot.graphed(src, library="ak")
    rec = gak.zip({"x": g.x, "doubled": g.x * 2.0 + 1.0})
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(rec, outdir, steps_per_file=4, tree_name="events")

    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    ref = uproot.open(src).arrays(["x"])
    assert (
        "doubled" in back.fields
    ), f"derived column dropped; wrote {sorted(back.fields)}"
    assert ak.array_equal(back["doubled"], ref.x * 2.0 + 1.0)
    # control: a plain-value field written beside the derived one
    assert ak.array_equal(back["x"], ref.x)


def test_pure_derived_record_is_named_by_its_field(tmp_path):
    """A record whose ONLY field is derived (``energy = sqrt(x^2 + y^2)``): the output tree carries
    the derived field name, not the raw source branches the expression reads."""
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
    """A derived per-object kinematic (``Jet_pt2 = Jet_pt * 2``) over a jagged branch: computed,
    with its list structure preserved."""
    src, jet = _make_jagged_root(os.path.join(tmp_path, "in.root"))
    g = uproot.graphed(src, library="ak")
    rec = gak.zip({"Jet_pt": g.Jet_pt, "Jet_pt2": g.Jet_pt * 2.0})
    outdir = os.path.join(tmp_path, "out")

    paths = uproot.graphed_write(rec, outdir, steps_per_file=1, tree_name="Events")

    back = ak.concatenate([uproot.open(p + ":Events").arrays() for p in paths])
    assert (
        "Jet_pt2" in back.fields
    ), f"derived jagged column dropped; wrote {sorted(back.fields)}"
    assert ak.array_equal(back["Jet_pt2"], jet * 2.0)
    assert ak.array_equal(back["Jet_pt"], jet)
