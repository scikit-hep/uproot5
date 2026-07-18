# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Awkward behaviors over real TTrees (graphed parity follow-up, item P1.2 / graphed M18).

``uproot.graphed(..., behavior=...)`` forwards a behavior dict to the recording backend, so
``gak.zip`` + ``gak.with_name`` give four-vector PROPERTIES (vector's ``Momentum4D`` — the coffea
pattern) on TTree branches: typetracer forms at record time, exact evaluation, and the projection
truthfulness pin — ``.pt`` reads exactly the px/py branches, which ``resolve_read_branches``
then serves minimally.
"""

import os

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")
vector = pytest.importorskip("vector")
vector.register_awkward()

from graphed.awkward import gak  # noqa: E402

BEHAVIOR = vector.backends.awkward.behavior


@pytest.fixture
def kinematics_file(tmp_path):
    rng = np.random.default_rng(7)
    n = 50
    cols = {
        "px": rng.normal(0.0, 10.0, n),
        "py": rng.normal(0.0, 10.0, n),
        "pz": rng.normal(0.0, 20.0, n),
        "E": rng.uniform(30.0, 60.0, n),
        "unused": rng.normal(size=n),
    }
    path = os.path.join(tmp_path, "kin.root")
    with uproot.recreate(path) as f:
        f["events"] = cols
    return path + ":events", cols


def _vectors(g):
    v = gak.zip({"px": g.px, "py": g.py, "pz": g.pz, "E": g.E})
    return gak.with_name(v, "Momentum4D")


def _reference(cols):
    """The SAME vector computation, eagerly — exact down to the ULP (hand formulas are not)."""
    return ak.Array(
        ak.zip({k: cols[k] for k in ("px", "py", "pz", "E")}, with_name="Momentum4D").layout,
        behavior=BEHAVIOR,
    )


def test_behavior_properties_record_and_evaluate(kinematics_file):
    where, cols = kinematics_file
    g = uproot.graphed(where, library="ak", behavior=BEHAVIOR)
    v = _vectors(g)
    pt = v.pt  # a behavior PROPERTY, not a TTree branch
    assert g.session.form(pt).is_typetracer  # inferred at record time, metadata only
    ref = _reference(cols)
    got = ak.Array(g.session.materialize(pt))
    assert ak.array_equal(got, ref.pt)  # exact: the reference IS vector's own computation
    mass = ak.Array(g.session.materialize(v.mass))
    assert ak.array_equal(mass, ref.mass, equal_nan=True)


def test_behavior_properties_project_to_their_branches(kinematics_file):
    where, _ = kinematics_file
    g = uproot.graphed(where, library="ak", behavior=BEHAVIOR)
    pt = _vectors(g).pt
    # the truthfulness pin: pt = hypot(px, py) reads EXACTLY those branches — never pz/E/unused
    assert uproot.necessary_columns(pt) == {"events": frozenset({"px", "py"})}


def test_unknown_attributes_fail_at_record_time(kinematics_file):
    # NOTE: vector.register_awkward() installs GLOBAL behaviors, so .pt works even without the
    # explicit kwarg (awkward semantics); what must still fail loudly is a NONEXISTENT property
    from graphed import GraphedTypeError

    where, _ = kinematics_file
    g = uproot.graphed(where, library="ak", behavior=BEHAVIOR)
    v = _vectors(g)
    with pytest.raises(GraphedTypeError):
        _ = v.definitely_not_a_property


def test_behavior_flows_through_the_generic_to_parquet(kinematics_file, tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("graphed_exec_local")
    import graphed.awkward.io as gio
    from graphed_exec_local import ProcessExecutor

    where, cols = kinematics_file
    g = uproot.graphed(where, library="ak", behavior=BEHAVIOR)
    pt = _vectors(g).pt
    # process workers cannot pickle vector's behavior dict (it contains lambdas): pass an
    # IMPORTABLE module:attr reference, resolved in each worker (the OpSpec pattern)
    paths = gio.to_parquet(
        pt, os.path.join(tmp_path, "pt"), steps_per_file=2,
        behavior="vector.backends.awkward:behavior",
        executor=ProcessExecutor(max_workers=2),
    )
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    assert ak.array_equal(back["data"], _reference(cols).pt)
