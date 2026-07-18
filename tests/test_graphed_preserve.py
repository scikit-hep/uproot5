# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""graphed-preserve bundle for a uproot-read graphed analysis, plus re-targeting the *preserved plan*
at alternate inputs.

Two complementary graphed preservation models:
  * a **Bundle** (M9) embeds the input data + canonical IR so ``reproduce`` recomputes the histogram
    bit-for-bit from references alone (clean machine, no original files), and ``inspect`` renders it
    without executing;
  * a **DurablePlan** (M8) preserves just the analysis graph (canonical IR), so ``with_partitions``
    re-targets it at a different file location or a different number of partitions without
    re-recording — "compile once, run on N datasets".
"""

import os
import sys

import awkward as ak
import numpy as np
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.preserve")
pytest.importorskip("graphed.awkward")

sys.path.insert(0, os.path.dirname(__file__))
from graphed import Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.checkpoint import Store, run_resumable
from graphed.preserve import Bundle, UnresolvedPayload, build_bundle, inspect, reproduce

HIST = {"name": "sum_pxpy", "bins": 50, "lo": -200.0, "hi": 200.0}
COLUMNS = ["px1", "py1"]


def _events(path_with_tree):
    return uproot.open(path_with_tree).arrays(COLUMNS, library="ak")


def _record(events):
    s = Session(AwkwardBackend())
    ev = from_awkward(s, "events", events)
    value = ev.px1 + ev.py1
    weight = ev.px1 * 0.0 + 1.0  # unit weights
    return s, value, weight


def _reference(events):
    v = ak.to_numpy(events.px1 + events.py1).astype("float64")
    counts, _ = np.histogram(
        v, bins=HIST["bins"], range=(HIST["lo"], HIST["hi"]), weights=np.ones(len(v))
    )
    return np.round(counts, 6)


def _build(root, path_with_tree):
    events = _events(path_with_tree)
    s, value, weight = _record(events)
    bundle = build_bundle(
        root, session=s, value=value, weight=weight,
        datasets={"events": events}, payloads={}, histogram=HIST,
    )
    return bundle, events


# ---- the Bundle: reproduce / inspect / fingerprint ----------------------------------------------
def test_reproduce_matches_build_bit_for_bit(tmp_path):
    p = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    bundle, events = _build(tmp_path / "b", p)
    ref = _reference(events)

    assert int(ref.sum()) > 0  # non-vacuous
    out = reproduce(bundle)
    assert np.array_equal(out, ref)  # reproduce (from references) == direct computation
    assert np.array_equal(reproduce(bundle), out)  # deterministic


def test_reproduce_from_clean_reload_without_original_file(tmp_path):
    p = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    bundle, events = _build(tmp_path / "b", p)
    ref = _reference(events)

    # reload from disk alone (no Session, no uproot file): the data is embedded in the bundle
    reloaded = Bundle.open(tmp_path / "b")
    assert np.array_equal(reproduce(reloaded), ref)


def test_inspect_renders_without_executing_and_missing_data_is_caught(tmp_path):
    p = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    bundle, _ = _build(tmp_path / "b", p)

    text = inspect(bundle)
    assert bundle.fingerprint() in text
    assert "Preservation Bundle" in text

    # inspect needs no event data: it still renders after the embedded dataset is removed,
    # while reproduce then fails with a precise UnresolvedPayload
    src_hash = bundle.manifest["sources"]["events"]
    (bundle.root / "store" / "objects" / src_hash).unlink()
    reloaded = Bundle.open(tmp_path / "b")
    assert reloaded.fingerprint() in inspect(reloaded)
    with pytest.raises(UnresolvedPayload):
        reproduce(reloaded)


def test_self_fingerprinting(tmp_path):
    p = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    events = _events(p)

    def _bundle(root, ev):
        s, value, weight = _record(ev)
        return build_bundle(
            root, session=s, value=value, weight=weight,
            datasets={"events": ev}, payloads={}, histogram=HIST,
        )

    same_a = _bundle(tmp_path / "a", events)
    same_b = _bundle(tmp_path / "b", events)
    assert same_a.fingerprint() == same_b.fingerprint()  # identical input -> identical fingerprint

    subset = events[:500]
    different = _bundle(tmp_path / "c", subset)
    assert different.fingerprint() != same_a.fingerprint()  # different data -> different fingerprint
    assert np.array_equal(reproduce(different), _reference(subset))


# ---- the preserved plan re-targeted at alternate inputs -----------------------------------------
def test_preserved_plan_runs_on_an_alternate_file(tmp_path):
    import graphed_uproot_report as gr

    p_a = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    p_b = skhep_testdata.data_path("uproot-Zmumu-uncompressed.root") + ":events"  # a different location

    base = gr.build_plan([t.partition for t in uproot.graphed_partitions(p_a, steps_per_file=4)])
    # re-target the SAME preserved analysis at a different file — the IR is shared, not re-recorded
    plan_b = base.with_partitions(
        [t.partition for t in uproot.graphed_partitions(p_b, steps_per_file=4)]
    )
    assert plan_b.ir == base.ir  # compile once

    result = run_resumable(plan_b, Store(str(tmp_path / "b"))).value
    assert np.array_equal(result, gr.single_pass(p_b))


def test_preserved_plan_runs_with_a_different_partition_count(tmp_path):
    import graphed_uproot_report as gr

    p = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    base = gr.build_plan([t.partition for t in uproot.graphed_partitions(p, steps_per_file=4)])
    re_chunked = base.with_partitions(
        [t.partition for t in uproot.graphed_partitions(p, steps_per_file=9)]
    )
    assert re_chunked.ir == base.ir

    r4 = run_resumable(base, Store(str(tmp_path / "n4"))).value
    r9 = run_resumable(re_chunked, Store(str(tmp_path / "n9"))).value
    assert np.array_equal(r4, r9)  # partition count does not change the result
    assert np.array_equal(r4, gr.single_pass(p))
