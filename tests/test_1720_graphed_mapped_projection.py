# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""The read list of a form-mapped ``uproot.graphed`` source.

With a mapping in place the graph's field names are not the file's branch names, so the syntactic
read list every driver computes today is meaningless for this source — it would ask the file for
``Jet``. The mapped source therefore DECLARES its own read list (``projected_columns``, which
``graphed.write.declared_columns`` asks for driver-side), computed from graphed's buffer-level
projection through the mapping's own ``keys_for_buffer_keys``.

The guard is a count-only output: ``gak.num(events.Jet, axis=1)`` reads no ``Jet_*`` branch at all,
only the counter branch ``nJet`` that appears nowhere in the graph. The helpers
(``necessary_columns`` / ``necessary_buffers`` / ``graphed_head``) and the generic writers answer
in the same ``TBranch`` names.
"""

import os

import awkward as ak
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")
graphed_write = pytest.importorskip("graphed.write")

if not hasattr(graphed_write, "declared_columns"):
    pytest.skip(
        "graphed.write.declared_columns (source-declared read lists) is required",
        allow_module_level=True,
    )

import graphed  # noqa: E402
from graphed import BufferNeed  # noqa: E402
from graphed.awkward import gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed import mini_nanoaod as mini  # noqa: E402


def _nano():
    return skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"


def _mapped():
    return uproot.graphed(
        _nano(),
        library="ak",
        filter_name=mini.FILTER,
        form_mapping=mini.MiniNanoMapping(),
    )


def _source_of(array):
    ((_nid, src),) = array.session.sources().items()
    return src


def test_count_only_output_declares_the_counter_branch():
    g = _mapped()
    n_jet = gak.num(g.Jet, axis=1)  # needs list STRUCTURE only: no Jet_* branch
    sum_pt = gak.sum(g.Jet.pt, axis=1)
    source = _source_of(g)

    assert source.projected_columns((n_jet,)) == ("nJet",)
    # across BOTH outputs: the counter plus exactly the one payload branch, nothing else
    assert set(graphed_write.declared_columns(source, (n_jet, sum_pt))) == {
        "nJet",
        "Jet_pt",
    }

    seen = []
    real_read = source.read_partition
    source.read_partition = lambda part, cols, res: (
        seen.append(tuple(cols)),
        real_read(part, cols, res),
    )[1]
    plan = graphed.aggregate_plan(
        n_jet,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        steps_per_file=2,
    )
    assert plan.process.columns == ("nJet",)
    total = SequentialRunner().run(plan).value
    assert seen == [("nJet",), ("nJet",)]  # every partition read only the counter

    events = mini.eager_events(_nano())
    assert total == float(ak.sum(ak.num(events.Jet, axis=1)))


def test_external_node_still_gets_a_read_list():
    g = _mapped()
    external = graphed.apply(lambda x: x * 0.5, gak.sum(g.Jet.pt, axis=1), name="halve")
    source = _source_of(g)

    # graphed's DEFAULT projection policy refuses to see through an External at all ...
    with pytest.raises(Exception):  # noqa: B017  (graphed's ProjectionError)
        uproot.necessary_columns(external)
    # ... so the declaring hook must not inherit it: the External's inputs are still read
    assert set(graphed_write.declared_columns(source, (external,))) == {
        "nJet",
        "Jet_pt",
    }

    plan = graphed.aggregate_plan(
        external,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        steps_per_file=2,
    )
    events = mini.eager_events(_nano())
    assert SequentialRunner().run(plan).value == 0.5 * float(
        ak.sum(ak.sum(events.Jet.pt, axis=1))
    )


def test_helpers_answer_in_tbranch_names_for_a_mapped_source():
    g = _mapped()
    sum_pt = gak.sum(g.Jet.pt, axis=1)
    assert uproot.necessary_columns(sum_pt) == {"Events": frozenset({"Jet_pt", "nJet"})}
    assert set(uproot.necessary_columns(sum_pt)["Events"]) == set(
        _source_of(g).projected_columns((sum_pt,))
    )

    # after mapping, a list's structure IS a branch read (its counter), so the buffer view is
    # branch-named too and composes with resolve_read_branches
    needs = uproot.necessary_buffers(gak.num(g.Jet, axis=1))
    assert needs == {"Events": {"nJet": BufferNeed.DATA}}
    tree = uproot.open(_nano())
    assert uproot.resolve_read_branches(tree, needs["Events"]) == {"nJet": "nJet"}


def test_graphed_head_peeks_through_the_mapping():
    g = _mapped()
    events = mini.eager_events(_nano())
    got = ak.Array(uproot.graphed_head(gak.sum(g.Jet.pt, axis=1), 5))
    assert len(got) == 5
    assert ak.array_equal(got, ak.sum(events.Jet.pt, axis=1)[:5])
    # a count-only peek can only produce these numbers by reading the counter branch: nothing
    # named Jet exists in the file, and the graph names no branch at all
    counted = ak.Array(uproot.graphed_head(gak.num(g.Jet, axis=1), 5))
    assert ak.array_equal(counted, ak.num(events.Jet, axis=1)[:5])


def test_to_parquet_reads_only_the_declared_branches(tmp_path):
    pytest.importorskip("pyarrow")
    import graphed.awkward.io as gio

    g = _mapped()
    source = _source_of(g)
    seen = []
    real_read = source.read_partition
    source.read_partition = lambda part, cols, res: (
        seen.append(tuple(cols)),
        real_read(part, cols, res),
    )[1]

    paths = gio.to_parquet(
        gak.num(g.Jet, axis=1), os.path.join(tmp_path, "out"), steps_per_file=2
    )
    assert seen == [("nJet",), ("nJet",)]
    back = ak.concatenate([ak.from_parquet(p) for p in paths])
    events = mini.eager_events(_nano())
    assert ak.array_equal(back["data"], ak.num(events.Jet, axis=1))
