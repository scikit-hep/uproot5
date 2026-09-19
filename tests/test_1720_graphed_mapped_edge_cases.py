# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""Edges of the form-mapped ``uproot.graphed`` source: an empty read list, a source named without
opening a file, a mapped form that is not a record, and behavior on arrays the graph builds.
"""

import awkward as ak
import numpy as np
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
from graphed.awkward import gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed import mini_nanoaod as mini  # noqa: E402


def _nano():
    return skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"


def _mapped(**kwargs):
    kwargs.setdefault("form_mapping", mini.MiniNanoMapping())
    return uproot.graphed(_nano(), library="ak", filter_name=mini.FILTER, **kwargs)


def _source_of(array):
    ((_nid, src),) = array.session.sources().items()
    return src


def _watch_reads(source):
    """Record the columns every partition read actually asks the file for."""
    seen = []
    real_read = source.read_range
    source.read_range = lambda tree, cols, start, stop: (
        seen.append(tuple(cols)),
        real_read(tree, cols, start, stop),
    )[1]
    return seen


def _sum64(values):
    """Sum in float64, so a two-partition total and a whole-file one agree to 1e-9."""
    return float(np.asarray(values).sum(dtype=np.float64))


def _total(output, steps_per_file=2):
    plan = graphed.aggregate_plan(
        output,
        reduce=lambda vals: _sum64(vals[0]),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        steps_per_file=steps_per_file,
    )
    return plan, SequentialRunner().run(plan).value


def test_an_empty_declared_read_list_reads_no_branch():
    # a record's LENGTH needs no buffer, so the declaration is empty — and an empty declaration
    # is an answer ("nothing"), not a missing one ("everything")
    g = _mapped()
    source = _source_of(g)
    n_events = gak.num(g, axis=0)
    assert source.projected_columns((n_events,)) == ()

    seen = _watch_reads(source)
    plan, total = _total(n_events)
    assert plan.process.columns == ()
    assert seen == [(), ()]  # every partition read no branch at all
    assert total == float(len(mini.eager_events(_nano())))


def test_a_nonempty_declared_read_list_is_still_honoured():
    # CONTROL for the empty-read-list case: a declaration that names branches still reaches the read
    g = _mapped()
    source = _source_of(g)
    seen = _watch_reads(source)
    _plan, total = _total(gak.sum(g.Jet.pt, axis=1))
    assert seen == [("Jet_pt", "nJet"), ("Jet_pt", "nJet")]
    events = mini.eager_events(_nano())
    assert total == pytest.approx(_sum64(ak.sum(events.Jet.pt, axis=1)), rel=1e-9)


def test_known_base_form_keeps_the_trees_name():
    mapping = mini.MiniNanoMapping()
    opened = _mapped(form_mapping=mapping)
    # the file does not exist, so this construction demonstrably opened nothing — yet the source
    # is named for the TTree, because the name is in the object path the caller gave
    blind = uproot.graphed(
        "/uproot-graphed/no-such-file.root:Events",
        library="ak",
        form_mapping=mini.MiniNanoMapping(),
        known_base_form=mapping.seen_base_form,
    )
    assert blind.session.source_name(0) == opened.session.source_name(0) == "Events"
    assert uproot.necessary_columns(gak.sum(blind.Jet.pt, axis=1)) == (
        uproot.necessary_columns(gak.sum(opened.Jet.pt, axis=1))
    )


class ScalarMapping:
    """A mapping whose mapped form is a bare ``NumpyForm``: one scalar branch, no record at all."""

    behavior = None

    def __call__(self, base_form):
        return base_form.content("MET_pt").copy(form_key="MET_pt"), self

    @property
    def buffer_key(self):
        return lambda form_key, form, attribute: f"/{attribute}/{form_key}"

    def keys_for_buffer_keys(self, buffer_keys):
        return frozenset(key.rsplit("/", 1)[1] for key in buffer_keys)

    def load_buffers(self, tree, keys, start, stop, _dec, _interp, _options):
        arrays = tree.arrays(list(keys), entry_start=start, entry_stop=stop)
        return {f"/data/{key}": ak.to_numpy(arrays[key]) for key in keys}


def test_a_mapped_form_that_is_not_a_record_projects_and_reads():
    g = uproot.graphed(
        _nano(), library="ak", filter_name=["MET_pt"], form_mapping=ScalarMapping()
    )
    assert _source_of(g).projected_columns((g,)) == ("MET_pt",)
    _plan, total = _total(g)
    met = uproot.open(_nano())["MET_pt"].array()
    assert total == pytest.approx(_sum64(met), rel=1e-9)


def test_the_mappings_behavior_reaches_arrays_the_graph_builds():
    # the chunks the mapping builds carry its behavior themselves; a record the GRAPH builds gets
    # it only from the backend, so `pt2` here is the behavior arriving through uproot.graphed
    g = _mapped()
    jets = gak.zip({"pt": g.Jet.pt}, with_name="MiniJet")
    _plan, total = _total(gak.sum(jets.pt2, axis=1))
    events = mini.eager_events(_nano())
    assert total == pytest.approx(_sum64(ak.sum(events.Jet.pt2, axis=1)), rel=1e-9)


class _SeesTheBaseForm:
    """A form mapping that only records the base form it is handed."""

    def __call__(self, form):
        self.form = form
        raise _Seen


class _Seen(Exception):
    pass


@pytest.mark.parametrize("make", ["mktree", "mkrntuple"])
def test_the_base_form_carries_the_requested_docs_for_trees_and_rntuples(
    tmp_path, make
):
    # a mapping (coffea's reads ``parameters["__doc__"]``) sees the same base form either way
    path = str(tmp_path / "docs.root")
    with uproot.recreate(path) as f:
        getattr(f, make)("events", {"x": np.float64, "n": np.int64})
    mapping = _SeesTheBaseForm()
    with pytest.raises(_Seen):
        uproot.graphed(
            f"{path}:events",
            library="ak",
            form_mapping=mapping,
            ak_add_doc={"__doc__": "title"},
        )
    assert all("__doc__" in mapping.form.content(name).parameters for name in "xn")
