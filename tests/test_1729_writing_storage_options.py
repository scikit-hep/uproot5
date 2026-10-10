# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
uproot.create, uproot.recreate and uproot.update document that extra keyword
arguments are passed to fsspec as storage_options, but any of them raised TypeError.
uproot.dask_write also did not pass its storage_options to the files it writes.
"""

from __future__ import annotations

import os

import awkward as ak
import fsspec.core
import pytest

import uproot


@pytest.fixture
def url_to_fs_kwargs(monkeypatch):
    """Record the keyword arguments of every fsspec.core.url_to_fs call."""
    seen = []
    url_to_fs = fsspec.core.url_to_fs

    def recording_url_to_fs(url, **kwargs):
        seen.append(kwargs)
        return url_to_fs(url, **kwargs)

    monkeypatch.setattr(fsspec.core, "url_to_fs", recording_url_to_fs)
    return seen


@pytest.mark.parametrize("function", [uproot.create, uproot.recreate])
def test_create_recreate_storage_options(tmp_path, url_to_fs_kwargs, function):
    path = os.path.join(tmp_path, "file.root")
    with function(path, auto_mkdir=True, compression=None) as file:
        file["x"] = "hello"

    # the storage option reaches the filesystem, the uproot option does not
    assert url_to_fs_kwargs
    assert all(kwargs == {"auto_mkdir": True} for kwargs in url_to_fs_kwargs)
    with uproot.open(path) as file:
        assert file.keys() == ["x;1"]


def test_update_storage_options(tmp_path, url_to_fs_kwargs):
    path = os.path.join(tmp_path, "file.root")
    with uproot.recreate(path) as file:
        file["x"] = "hello"
    url_to_fs_kwargs.clear()

    with uproot.update(path, auto_mkdir=True, initial_directory_bytes=512) as file:
        file["y"] = "world"

    assert url_to_fs_kwargs
    assert all(kwargs == {"auto_mkdir": True} for kwargs in url_to_fs_kwargs)
    with uproot.open(path) as file:
        assert sorted(file.keys()) == ["x;1", "y;1"]


@pytest.mark.parametrize("function", [uproot.create, uproot.recreate, uproot.update])
def test_misspelled_option_is_rejected(tmp_path, url_to_fs_kwargs, function):
    path = os.path.join(tmp_path, "file.root")
    # not handed to the filesystem, which may silently ignore it
    with pytest.raises(
        TypeError, match=r"'compresion' \(did you mean 'compression'\?\)"
    ):
        function(path, compresion=None)
    assert url_to_fs_kwargs == []
    assert not os.path.exists(path)


def test_update_rejects_create_only_option(tmp_path, url_to_fs_kwargs):
    path = os.path.join(tmp_path, "file.root")
    with uproot.recreate(path) as file:
        file["x"] = "hello"
    url_to_fs_kwargs.clear()

    # compression is an uproot option, but not one uproot.update takes
    with pytest.raises(TypeError, match=r"unrecognized options for uproot\.update"):
        uproot.update(path, compression=None)
    assert url_to_fs_kwargs == []


def test_dask_write_forwards_storage_options(tmp_path, monkeypatch):
    dask_awkward = pytest.importorskip("dask_awkward")

    seen = []
    recreate = uproot.recreate

    def recording_recreate(file_path, **options):
        seen.append(options)
        return recreate(file_path, **options)

    monkeypatch.setattr(uproot, "recreate", recording_recreate)

    arr = ak.Array([{"a": [1, 2, 3]}, {"a": [4, 5]}])
    dask_arr = dask_awkward.from_awkward(arr, npartitions=2)
    out = uproot.dask_write(
        dask_arr,
        str(tmp_path),
        prefix="data",
        storage_options={"auto_mkdir": True},
        compute=False,
    )
    # in this process, so that the monkeypatch is visible to the writing tasks
    out.compute(scheduler="sync")

    assert len(seen) == 2
    assert all(options.get("auto_mkdir") is True for options in seen)
    with uproot.open(os.path.join(tmp_path, "data-part0.root")) as file:
        assert file["tree"].num_entries == 1
