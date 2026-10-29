# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import uproot

graphed = pytest.importorskip("graphed")
pytest.importorskip("graphed.awkward")
from graphed.awkward import AwkwardBackend  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(graphed, "current_session"),
    reason="graphed has no one-binding-per-name rule",
)


@pytest.fixture
def two_cycles(tmp_path):
    """``events;1`` with 3 entries and ``events;2`` with 5, plus a second tree ``other``."""
    path = str(tmp_path / "cycles.root")
    with uproot.recreate(path) as file:
        file.mktree("events", {"x": np.int32}).extend(
            {"x": np.arange(3, dtype=np.int32)}
        )
        file.mktree("other", {"x": np.int32}).extend(
            {"x": np.arange(3, dtype=np.int32)}
        )
    with uproot.update(path) as file:
        file.mktree("events", {"x": np.int32}).extend(
            {"x": np.arange(5, dtype=np.int32)}
        )
    return path


def _read(path, object_path, session, **options):
    return uproot.graphed(
        {path: object_path}, session=session, name="events", **options
    )


@pytest.mark.parametrize(
    "opened, known",
    [("events", "events"), ("events", "/events"), ("events;1", "events;1")],
)
@pytest.mark.parametrize("known_first", [False, True])
def test_a_known_form_read_reuses_the_opened_reads_source(
    two_cycles, opened, known, known_first
):
    session = graphed.Session(AwkwardBackend())
    form = _read(two_cycles, opened, graphed.Session(AwkwardBackend()))
    form = form.session.form(form).tt.layout.form
    reads = [
        lambda: _read(two_cycles, opened, session),
        lambda: _read(two_cycles, known, session, known_base_form=form),
    ]
    first, second = reads[::-1] if known_first else reads
    assert first().node_id == second().node_id


@pytest.mark.parametrize(
    "held, other",
    [("events", "events;1"), ("events;1", "events"), ("events", "other")],
)
@pytest.mark.parametrize("known", [False, True])
def test_another_cycle_or_tree_is_refused(two_cycles, held, other, known):
    session = graphed.Session(AwkwardBackend())
    first = _read(two_cycles, held, session)
    options = {"known_base_form": session.form(first).tt.layout.form} if known else {}
    with pytest.raises(ValueError, match="already bound to other data"):
        _read(two_cycles, other, session, **options)
