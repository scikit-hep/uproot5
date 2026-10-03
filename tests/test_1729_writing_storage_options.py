# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
Extra keyword arguments to uproot.create, uproot.recreate and uproot.update are
documented as fsspec storage_options, but used to raise TypeError as unrecognized
options. uproot.dask_write also did not pass its storage_options to the files it
writes.
"""

from __future__ import annotations

import os

import awkward as ak
import pytest

import uproot


def test_recreate_storage_options(tmp_path):
    path = os.path.join(tmp_path, "sub", "dir", "file.root")
    with uproot.recreate(path, auto_mkdir=True) as file:
        file["x"] = "hello"
    with uproot.open(path) as file:
        assert file.keys() == ["x;1"]


def test_create_storage_options(tmp_path):
    path = os.path.join(tmp_path, "sub", "file.root")
    with uproot.create(path, auto_mkdir=True) as file:
        file["x"] = "hello"
    with uproot.open(path) as file:
        assert file.keys() == ["x;1"]


def test_update_storage_options(tmp_path):
    path = os.path.join(tmp_path, "file.root")
    with uproot.recreate(path) as file:
        file["x"] = "hello"
    with uproot.update(path, auto_mkdir=True) as file:
        file["y"] = "world"
    with uproot.open(path) as file:
        assert sorted(file.keys()) == ["x;1", "y;1"]


def test_known_options_still_validated(tmp_path):
    path = os.path.join(tmp_path, "file.root")
    with uproot.recreate(path) as file:
        file["x"] = "hello"
    # create-only options are not storage options, so update still rejects them
    with pytest.raises(TypeError, match=r"unrecognized options for uproot.update"):
        uproot.update(path, compression=None)


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
    uproot.dask_write(
        dask_arr,
        str(tmp_path),
        prefix="data",
        storage_options={"auto_mkdir": True},
        compute=True,
    )

    assert len(seen) == 2
    assert all(options.get("auto_mkdir") is True for options in seen)
    with uproot.open(os.path.join(tmp_path, "data-part0.root")) as file:
        assert file["tree"].num_entries == 1
