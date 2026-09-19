# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
"""``uproot.graphed_write`` over a form-mapped source, and the ``behavior=`` guard.

The write path evaluates the recorded graph per partition exactly as the read path does, so it
reads what the MAPPED source declares — ``TBranch`` names — and not the graph's field names, which
name nothing in the file.

``behavior=`` is the plain read's knob: ``backend=`` brings its own behavior and ``form_mapping=``
carries the mapping info's, so either beside it is a ``TypeError`` rather than a silent choice.
"""

import os

import awkward as ak
import pytest
import skhep_testdata

import uproot

pytest.importorskip("graphed.awkward")
pytest.importorskip("graphed_executors.local")
graphed_write = pytest.importorskip("graphed.write")

if not hasattr(graphed_write, "declared_columns"):
    pytest.skip(
        "graphed.write.declared_columns (source-declared read lists) is required",
        allow_module_level=True,
    )

from graphed.awkward import AwkwardBackend, gak  # noqa: E402

from tests.graphed import mini_nanoaod as mini  # noqa: E402


def _nano():
    return skhep_testdata.data_path("nanoAOD_2015_CMS_Open_Data_ttbar.root") + ":Events"


def test_writes_a_mapped_source_reading_only_the_declared_branches(tmp_path):
    g = uproot.graphed(
        _nano(),
        library="ak",
        filter_name=mini.FILTER,
        form_mapping=mini.MiniNanoMapping(),
    )
    ((_nid, source),) = g.session.sources().items()
    seen = []
    real_read = source.read_range
    source.read_range = lambda tree, cols, start, stop: (
        seen.append(tuple(cols)),
        real_read(tree, cols, start, stop),
    )[1]

    out = gak.zip(
        {"n_jet": gak.num(g.Jet, axis=1), "sum_pt": gak.sum(g.Jet.pt, axis=1)}
    )
    paths = uproot.graphed_write(
        out,
        os.path.join(tmp_path, "out"),
        steps_per_file=2,
        tree_name="events",
        executor="thread",
    )

    assert seen == [("Jet_pt", "nJet"), ("Jet_pt", "nJet")]
    events = mini.eager_events(_nano())
    back = ak.concatenate([uproot.open(p + ":events").arrays() for p in paths])
    assert ak.array_equal(back.n_jet, ak.num(events.Jet, axis=1))
    assert ak.array_equal(back.sum_pt, ak.sum(events.Jet.pt, axis=1))


@pytest.mark.parametrize(
    "instead",
    [
        {"backend": AwkwardBackend(behavior=mini.BEHAVIOR)},
        {"form_mapping": mini.MiniNanoMapping()},
    ],
)
def test_behavior_is_refused_beside_backend_or_form_mapping(instead):
    with pytest.raises(TypeError, match="behavior="):
        uproot.graphed(
            _nano(),
            library="ak",
            filter_name=mini.FILTER,
            behavior=mini.BEHAVIOR,
            **instead,
        )
