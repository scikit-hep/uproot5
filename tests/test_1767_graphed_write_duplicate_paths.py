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


@pytest.mark.parametrize(
    "spelling",
    [
        lambda absp, relp: {absp: "events", relp: "events"},
        lambda absp, relp: [absp + ":events", relp + ":events"],
        lambda absp, relp: [absp, relp],
        lambda absp, relp: [absp + ":events", relp],
        lambda absp, relp: {Path(absp).as_uri(): "events", relp: "events"},
        lambda absp, relp: [Path(absp), relp],
        lambda absp, relp: {absp: "events", f"./{relp}": "events"},
    ],
    ids=[
        "dict-spelling",
        "colon-list",
        "bare-list",
        "colon-and-bare",
        "file-protocol-dict",
        "path-object",
        "dot-slash-rel",
    ],
)
def test_same_file_under_two_path_spellings_is_refused(tmp_path, monkeypatch, spelling):
    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    relp = "a.root"

    g = uproot.graphed(spelling(absp, relp))
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out1"), executor="thread")


def test_same_file_under_two_path_spellings_open_files_is_refused(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    relp = "a.root"

    g = uproot.graphed({absp: "events", relp: "events"}, open_files=True)
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out2"), executor="thread")


def test_same_file_under_two_path_spellings_known_base_form_is_refused(
    tmp_path, monkeypatch
):
    from uproot._graphed import _base_form_of

    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    relp = "a.root"

    tree = uproot.open(absp)["events"]
    base_form = _base_form_of(tree, ["x", "y"], False, None)

    g = uproot.graphed(
        {absp: "events", relp: "events"},
        known_base_form=base_form,
    )
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out3"), executor="thread")


@pytest.mark.parametrize(
    "paths",
    [
        lambda absp, relp: {absp: "events", relp: "/events"},
        lambda absp, relp: {absp: "events;1", relp: "events"},
        lambda absp, relp: {absp: "/events;1", relp: "/events"},
        lambda absp, relp: {absp: None, relp: "events"},
        lambda absp, relp: {absp: "events", relp: None},
        lambda absp, relp: {absp: None, relp: "/events;1"},
    ],
    ids=[
        "leading-slash",
        "cycle-suffix",
        "leading-slash-and-cycle",
        "none-and-name",
        "name-and-none",
        "none-and-slash-cycle",
    ],
)
def test_object_path_duplicate_spellings_known_base_form(tmp_path, monkeypatch, paths):
    from uproot._graphed import _base_form_of

    monkeypatch.chdir(tmp_path)
    absp = _make_tree(tmp_path / "a.root")
    relp = "a.root"

    tree = uproot.open(absp)["events"]
    base_form = _base_form_of(tree, ["x", "y"], False, None)

    g = uproot.graphed(paths(absp, relp), known_base_form=base_form)
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out_obj"), executor="thread")


def test_hard_link_is_refused(tmp_path):
    absp = _make_tree(tmp_path / "a.root")
    linkp = str(tmp_path / "a_hardlink.root")
    os.link(absp, linkp)

    g = uproot.graphed({absp: "events", linkp: "events"})
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out_link"), executor="thread")


def test_case_difference_on_case_insensitive_filesystem(tmp_path):
    absp = _make_tree(tmp_path / "a.root")
    upperp = str(tmp_path / "A.ROOT")
    if not os.path.exists(upperp):
        pytest.skip("Filesystem is case sensitive")

    g = uproot.graphed({absp: "events", upperp: "events"})
    with pytest.raises(ValueError, match="duplicate input"):
        uproot.graphed_write(g, str(tmp_path / "out_case"), executor="thread")


def test_different_trees_in_same_file_under_two_path_spellings_are_accepted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    a = tmp_path / "a.root"
    with uproot.recreate(a) as f:
        for name in ("events", "other"):
            f.mktree(name, {"x": "f8", "y": "f8"})
            f[name].extend({"x": np.arange(10, dtype="f8"), "y": np.zeros(10)})
    absp = str(a)
    relp = "a.root"

    g = uproot.graphed([absp + ":events", relp + ":other"])
    paths = uproot.graphed_write(g, str(tmp_path / "out4"), executor="thread")
    assert len(paths) == 2


def test_canonical_path_normalization(tmp_path):
    from uproot._util import canonical_object_path, canonical_path

    assert canonical_path(None) is None
    assert canonical_path(123) == 123

    local_p = tmp_path / "test.root"
    local_p.touch()
    expected_local = str(local_p.resolve())

    assert canonical_path(local_p) == expected_local
    assert canonical_path(str(local_p).encode()) == expected_local

    class BytesPathLike:
        def __init__(self, raw: bytes):
            self._raw = raw

        def __fspath__(self):
            return self._raw

    assert canonical_path(BytesPathLike(str(local_p).encode())) == expected_local

    file_url = local_p.as_uri()
    assert canonical_path(file_url) == expected_local
    assert canonical_path(local_p.as_uri()) == expected_local
    assert (
        canonical_path("file://somehost/tmp/test.root")
        == "file://somehost/tmp/test.root"
    )

    assert (
        canonical_path("http://example.com/foo/../bar.root")
        == "http://example.com/bar.root"
    )
    assert canonical_path("root://server.cern.ch//a/../b.root") == canonical_path(
        "root://server.cern.ch/b.root"
    )
    assert canonical_path("root://server.cern.ch//a.root") == canonical_path(
        "root://server.cern.ch/a.root"
    )
    assert "b.root" in canonical_path("root://server.cern.ch//a/../b.root")
    assert "a.root" in canonical_path("root://server.cern.ch//a.root")
    assert (
        canonical_path("root://server.cern.ch/a.root") == "root://server.cern.ch/a.root"
    )
    assert (
        canonical_path("http://User:Secret@example.com/a.root")
        == "http://User:Secret@example.com/a.root"
    )

    assert canonical_object_path("/events") == "/events;1"
    assert canonical_object_path("events;1") == "/events;1"
    assert canonical_object_path("/events;1") == "/events;1"
    assert canonical_object_path(None, default="events") == "/events;1"
    assert canonical_object_path("/dir/events;2") == "/dir/events;1"
