# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import uproot

pytest.importorskip("graphed.awkward")
pytest.importorskip("graphed_executors.local")


def _make_tree(path, n=10, name="events"):
    with uproot.recreate(path) as f:
        f.mktree(name, {"x": "f8", "y": "f8"})
        f[name].extend({"x": np.arange(n, dtype="f8"), "y": 2.0 * np.arange(n)})
    return str(path)


def _relative_path(path):
    return os.path.join(
        os.path.dirname(path),
        "..",
        os.path.basename(os.path.dirname(path)),
        os.path.basename(path),
    )


@pytest.mark.parametrize(
    "spelling",
    [
        lambda absp, relp: {absp: "events", relp: "events"},
        lambda absp, relp: [absp + ":events", relp + ":events"],
        lambda absp, relp: [absp, relp],
        lambda absp, relp: [absp + ":events", relp],
        lambda absp, relp: {Path(absp).as_uri(): "events", relp: "events"},
        lambda absp, relp: [Path(absp), relp],
    ],
    ids=[
        "dict-spelling",
        "colon-list",
        "bare-list",
        "colon-and-bare",
        "file-protocol-dict",
        "path-object",
    ],
)
def test_same_file_under_two_path_spellings_is_refused(tmp_path, spelling):
    absp = _make_tree(tmp_path / "a.root")
    relp = _relative_path(absp)

    g = uproot.graphed(spelling(absp, relp))
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out1"), executor="thread")


def test_same_file_under_two_path_spellings_open_files_is_refused(tmp_path):
    absp = _make_tree(tmp_path / "a.root")
    relp = _relative_path(absp)

    g = uproot.graphed({absp: "events", relp: "events"}, open_files=True)
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out2"), executor="thread")


def test_same_file_under_two_path_spellings_known_base_form_is_refused(tmp_path):
    from uproot._graphed import _base_form_of

    absp = _make_tree(tmp_path / "a.root")
    relp = _relative_path(absp)

    tree = uproot.open(absp)["events"]
    base_form = _base_form_of(tree, ["x", "y"], False, None)

    g = uproot.graphed(
        {absp: "events", relp: "events"},
        known_base_form=base_form,
    )
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out3"), executor="thread")


def test_different_trees_in_same_file_under_two_path_spellings_are_accepted(tmp_path):
    a = tmp_path / "a.root"
    with uproot.recreate(a) as f:
        for name in ("events", "other"):
            f.mktree(name, {"x": "f8", "y": "f8"})
            f[name].extend({"x": np.arange(10, dtype="f8"), "y": np.zeros(10)})
    absp = str(a)
    relp = _relative_path(absp)

    g = uproot.graphed([absp + ":events", relp + ":other"])
    paths = uproot.graphed_write(g, str(tmp_path / "out4"), executor="thread")
    assert len(paths) == 2


def test_canonical_path_normalization(tmp_path):
    from uproot._util import canonical_path

    assert canonical_path(None) is None
    assert canonical_path(123) == 123

    local_p = tmp_path / "test.root"
    assert canonical_path(local_p) == str(local_p.resolve())
    assert canonical_path(str(local_p).encode()) == str(local_p.resolve())

    file_url = f"file://{local_p}"
    assert canonical_path(file_url) == str(local_p.resolve())
    assert canonical_path(local_p.as_uri()) == str(local_p.resolve())
    assert canonical_path("file://somehost/tmp/test.root") is not None

    assert (
        canonical_path("http://example.com/foo/../bar.root")
        == "http://example.com/bar.root"
    )

    assert (
        canonical_path("root://server.cern.ch//a/../b.root")
        == "root://server.cern.ch//b.root"
    )
