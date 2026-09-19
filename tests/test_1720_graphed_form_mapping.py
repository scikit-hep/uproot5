# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed(form_mapping=, known_base_form=, backend=)`` — a schema'd source for graphed.

``uproot.dask`` lets a caller (coffea's ``NanoEventsFactory``) hand the reader a form mapping: an
object that translates the flat ``TTree`` form into the shape the user sees and knows how to fill
the mapped form's buffers from the file. These pin the same three knobs on ``uproot.graphed``:

- the mapping object is *the same object* ``uproot.dask`` takes, and produces the same form;
- a recorded analysis over the mapped source, executed partition-wise, equals the eager answer
  bit-for-bit;
- ``known_base_form=`` records without opening a file at all;
- ``backend=`` takes a graphed ``Backend`` instance, so a caller can install its own ``Array``
  subclass (the route coffea's behaviors need) instead of the default awkward backend.

The mapping is ``tests.graphed.mini_nanoaod`` — a real restructuring (``nJet`` + ``Jet_*`` ->
``Jet: var * MiniJet[...]``), because ``TrivialFormMapping`` cannot tell graph field names and
branch names apart.
"""

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
from graphed.awkward import AwkwardBackend, gak  # noqa: E402
from graphed.core import SequentialRunner  # noqa: E402

from tests.graphed import mini_nanoaod as mini  # noqa: E402

MAPPED_FORM = (
    "## * {Jet: var * MiniJet[eta: float32, pt: float32], "
    "Muon: var * MiniJet[pt: float32], MET_pt: float32}"
)


def _nano():
    return skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"


def _source_of(array):
    ((_nid, src),) = array.session.sources().items()
    return src


def test_mapped_form_and_behavior_match_uproot_dask():
    pytest.importorskip("dask_awkward")
    mapping = mini.MiniNanoMapping()
    g = uproot.graphed(
        _nano(), library="ak", filter_name=mini.FILTER, form_mapping=mapping
    )
    # the ONE mapping instance is accepted by both readers and yields the same user-visible form
    d = uproot.dask(
        _nano(),
        library="ak",
        filter_name=mini.FILTER,
        form_mapping=mapping,
        open_files=False,
    )
    assert g.session.form(g).describe() == MAPPED_FORM == str(d._meta.type)

    # the mapping info's behavior reached the recording backend: `pt2` is a behavior PROPERTY of
    # the mapped record, computed at record time from a form that stores only pt/eta
    assert g.session.form(g.Jet.pt2).describe() == "## * var * float32"
    got = ak.Array(g.session.materialize(g.Jet.pt2))
    assert ak.array_equal(got, mini.eager_events(_nano()).Jet.pt2)


def test_mapped_analysis_runs_partitioned_bit_for_bit():
    # two files x three steps: the mapped read path has to rebuild the schema form per chunk
    g = uproot.graphed(
        [_nano(), _nano()],
        library="ak",
        filter_name=mini.FILTER,
        form_mapping=mini.MiniNanoMapping(),
    )
    n_jet = gak.num(g.Jet, axis=1)
    sum_pt = gak.sum(g.Jet.pt, axis=1)
    plan = graphed.aggregate_plan(
        n_jet,
        sum_pt,
        reduce=lambda vals: (float(ak.sum(vals[0])), float(ak.sum(vals[1]))),
        combine=lambda a, b: (a[0] + b[0], a[1] + b[1]),
        empty=lambda: (0.0, 0.0),
        steps_per_file=3,
    )
    assert len(plan.tasks) == 6
    result = SequentialRunner().run(plan)
    assert result.n_partitions == 6

    events = mini.eager_events(_nano(), n_copies=2)
    assert result.value == (
        float(ak.sum(ak.num(events.Jet, axis=1))),
        float(ak.sum(ak.sum(events.Jet.pt, axis=1))),
    )


def test_known_base_form_records_without_opening_a_file():
    mapping = mini.MiniNanoMapping()
    opened = uproot.graphed(
        _nano(), library="ak", filter_name=mini.FILTER, form_mapping=mapping
    )
    base_form = mapping.seen_base_form  # what the opening pass handed the mapping

    missing = "/uproot-graphed/no-such-file.root:Events"
    blind = uproot.graphed(
        missing,
        library="ak",
        form_mapping=mini.MiniNanoMapping(),
        known_base_form=base_form,
    )
    # the file does not exist, so construction demonstrably opened nothing
    assert blind.session.form(blind).describe() == MAPPED_FORM
    assert (
        blind.session.form(blind).describe() == opened.session.form(opened).describe()
    )
    with pytest.raises(Exception):  # noqa: B017  (without the form, the file is needed)
        uproot.graphed(missing, library="ak", form_mapping=mini.MiniNanoMapping())


def test_backend_instance_installs_its_array_type():
    class TaggedArray(graphed.Array):
        pass

    class TaggedBackend(AwkwardBackend):
        def array_type(self):
            return TaggedArray

    backend = TaggedBackend(behavior=mini.BEHAVIOR)
    g = uproot.graphed(
        _nano(),
        library="ak",
        filter_name=mini.FILTER,
        form_mapping=mini.MiniNanoMapping(),
        backend=backend,
    )
    assert g.session.backend is backend
    assert type(g) is TaggedArray
    assert type(g.Jet.pt2) is TaggedArray  # recorded nodes come back as the subclass

    default = uproot.graphed(_nano(), library="ak", filter_name=mini.FILTER)
    assert type(default) is graphed.Array
    assert type(default.nJet) is graphed.Array


def test_dict_ak_add_doc_and_typenames_reach_the_mapping():
    # coffea's dask arm passes a DICT ak_add_doc and reads form.parameters["typenames"];
    # both have to survive into the mapping and out into the recorded form
    mapping = mini.MiniNanoMapping()
    g = uproot.graphed(
        _nano(),
        library="ak",
        filter_name=mini.FILTER,
        form_mapping=mapping,
        ak_add_doc={"__doc__": "title", "typename": "typename"},
    )
    assert mapping.seen_base_form.parameters["typenames"]["nJet"] == "uint32_t"
    recorded = g.session.form(g).tt.layout.form
    assert recorded.content("Jet").content.content("pt").parameters == {
        "__doc__": "pt",
        "typename": "float[]",
    }
    assert recorded.content("MET_pt").parameters == {
        "__doc__": "pt",
        "typename": "float",
    }


def test_unmapped_source_declares_no_read_list():
    # CONTROL: without form_mapping the source declares nothing and the plan keeps today's
    # syntactic read list, so the mapped path cannot silently change the unmapped one
    where = skhep_testdata.data_path("uproot-Zmumu.root") + ":events"
    g = uproot.graphed(where, library="ak")
    expr = g.px1 + g.py1
    assert graphed_write.declared_columns(_source_of(g), (expr,)) is None
    plan = graphed.aggregate_plan(
        expr,
        reduce=lambda vals: float(ak.sum(vals[0])),
        combine=lambda a, b: a + b,
        empty=lambda: 0.0,
        steps_per_file=2,
    )
    assert set(plan.process.columns) == {"px1", "py1"}
    raw = uproot.open(where).arrays(["px1", "py1"])
    assert SequentialRunner().run(plan).value == float(ak.sum(raw.px1 + raw.py1))
