# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""NanoAOD witnesses (ADL-port P0.2): the porting idiom over counted jagged TTree branches.

The ADL queries zip exactly the columns they need, name them, and lean on vector behaviors —
no schema layer. These witnesses pin that idiom over a REAL NanoAOD-style file (counted jagged
branches written with shared nJet/nMuon counters): form-correct recording, truthful projection
(a behavior property reads exactly its branches even when the zip names more), record+record
arithmetic through the proxy's `+`, jagged-integer-array getitem over the reader source, and
the capstone — TTree -> zipped collection -> behavior property -> a deferred hist.graphed fill
aggregated by a spawned process pool with the behavior forwarded by import ref.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import awkward as ak
import numpy as np
import pytest

import uproot

pytest.importorskip("graphed_awkward")
vector = pytest.importorskip("vector")
vector.register_awkward()

from graphed_awkward import gak  # noqa: E402

BEHAVIOR = vector.backends.awkward.behavior
N = 300


@pytest.fixture(scope="module")
def nanoaod_file(tmp_path_factory):
    """A NanoAOD-style TTree: shared-counter jagged collections + flat MET branches."""
    rng = np.random.default_rng(2012)
    nmu = rng.integers(0, 5, N)
    njet = rng.integers(0, 7, N)

    def kin(counts, scale):
        tot = int(counts.sum())
        return {
            "pt": ak.unflatten(rng.exponential(scale, tot) + 0.5, counts),
            "eta": ak.unflatten(rng.normal(0.0, 1.8, tot), counts),
            "phi": ak.unflatten(rng.uniform(-np.pi, np.pi, tot), counts),
            "mass": ak.unflatten(rng.uniform(0.1, 1.0, tot), counts),
        }

    muon = ak.zip({**kin(nmu, 25.0), "charge": ak.unflatten(rng.choice([-1, 1], int(nmu.sum())), nmu)})
    jet = ak.zip(kin(njet, 40.0))
    path = os.path.join(tmp_path_factory.mktemp("nano"), "nano.root")
    with uproot.recreate(path) as f:
        f.mktree("Events", {"Muon": muon.type, "Jet": jet.type, "MET_pt": "float64", "MET_phi": "float64"})
        f["Events"].extend({
            "Muon": muon, "Jet": jet,
            "MET_pt": rng.exponential(20.0, N), "MET_phi": rng.uniform(-np.pi, np.pi, N),
        })
    raw = uproot.open(path + ":Events").arrays()
    return path + ":Events", raw


def _muons(g):
    return gak.zip(
        {"pt": g.Muon_pt, "eta": g.Muon_eta, "phi": g.Muon_phi,
         "mass": g.Muon_mass, "charge": g.Muon_charge},
        with_name="Momentum4D",
    )


def _ref_muons(raw):
    return ak.zip(
        {"pt": raw.Muon_pt, "eta": raw.Muon_eta, "phi": raw.Muon_phi,
         "mass": raw.Muon_mass, "charge": raw.Muon_charge},
        with_name="Momentum4D", behavior=BEHAVIOR,
    )


def test_counted_jagged_zip_records_evaluates_and_projects_truthfully(nanoaod_file):
    where, raw = nanoaod_file
    g = uproot.graphed(where, library="ak", behavior=BEHAVIOR)
    muons = _muons(g)
    # form correct at record time: a jagged record of the five fields
    assert "var" in g.session.form(muons).describe()
    # a behavior property over the jagged collection, exact vs the SAME eager vector ops
    got = ak.Array(g.session.materialize(muons.px))
    assert ak.array_equal(got, _ref_muons(raw).px)
    # truthful projection: px touches ONLY pt/phi even though the zip names five branches
    assert uproot.necessary_columns(muons.px) == {"Events": frozenset({"Muon_pt", "Muon_phi"})}


def test_record_arithmetic_four_vector_sums_over_the_reader(nanoaod_file):
    where, raw = nanoaod_file
    g = uproot.graphed(where, library="ak", behavior=BEHAVIOR)
    pairs = gak.combinations(_muons(g), 2, fields=["a", "b"])
    mass = (pairs.a + pairs.b).mass  # record + record through the proxy's `+`
    ref_pairs = ak.combinations(_ref_muons(raw), 2, fields=["a", "b"])
    ref_mass = (ref_pairs.a + ref_pairs.b).mass
    assert ak.array_equal(
        ak.Array(g.session.materialize(mass)), ref_mass, equal_nan=True
    )


def test_jagged_integer_array_getitem_over_the_reader(nanoaod_file):
    where, raw = nanoaod_file
    g = uproot.graphed(where, library="ak", behavior=BEHAVIOR)
    muons = _muons(g)
    pair = gak.argcombinations(muons, 2, fields=["l1", "l2"])
    picked = muons[pair.l1]  # Q8's leptons[pair.l1]: jagged-integer-array indexing
    ref = _ref_muons(raw)
    ref_pair = ak.argcombinations(ref, 2, fields=["l1", "l2"])
    assert ak.array_equal(
        ak.Array(g.session.materialize(picked.pt)), ref[ref_pair.l1].pt
    )


def test_capstone_ttree_to_histogram_through_a_process_pool(nanoaod_file, tmp_path, monkeypatch):
    gh = pytest.importorskip("graphed_histogram")
    pytest.importorskip("graphed_exec_local")
    import boost_histogram as bh
    from graphed_exec_local import ProcessExecutor

    # spawned workers resolve the backend by IMPORT REF: the ref module lives beside this test,
    # so put the tests dir on sys.path (spawn children inherit it via the preparation data)
    monkeypatch.syspath_prepend(os.path.dirname(__file__))

    where, raw = nanoaod_file
    g = uproot.graphed(where, library="ak", behavior=BEHAVIOR)
    pt = _muons(g).pt
    h = gh.boost.Histogram(bh.axis.Regular(40, 0.0, 120.0), storage=bh.storage.Int64())
    h.fill(pt)
    plan = h.plan(steps_per_file=3, backend="vector_backend_ref:make_backend")
    out = ProcessExecutor(max_workers=2).run(plan).value
    eager = bh.Histogram(bh.axis.Regular(40, 0.0, 120.0), storage=bh.storage.Int64())
    eager.fill(ak.flatten(_ref_muons(raw).pt, axis=None))
    assert np.array_equal(np.asarray(out.values(flow=True)), eager.values(flow=True))
