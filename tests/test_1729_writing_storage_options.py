# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
uproot.create, uproot.recreate and uproot.update documented that extra keyword
arguments are passed to fsspec as storage_options, but any of them raised TypeError.
They now take an explicit storage_options argument, so a misspelled option is still
rejected. uproot.dask_write also did not pass its storage_options to the files it
writes.
"""

from __future__ import annotations

import os

import awkward as ak
import pytest

import uproot


def test_recreate_storage_options(tmp_path):
    path = os.path.join(tmp_path, "sub", "dir", "file.root")
    with uproot.recreate(path, storage_options={"auto_mkdir": True}) as file:
        file["x"] = "hello"
    with uproot.open(path) as file:
        assert file.keys() == ["x;1"]


def test_create_storage_options(tmp_path):
    path = os.path.join(tmp_path, "sub", "file.root")
    with uproot.create(path, storage_options={"auto_mkdir": True}) as file:
        file["x"] = "hello"
    with uproot.open(path) as file:
        assert file.keys() == ["x;1"]


def test_update_storage_options(tmp_path):
    path = os.path.join(tmp_path, "file.root")
    with uproot.recreate(path) as file:
        file["x"] = "hello"
    with uproot.update(path, storage_options={"auto_mkdir": True}) as file:
        file["y"] = "world"
    with uproot.open(path) as file:
        assert sorted(file.keys()) == ["x;1", "y;1"]


@pytest.mark.parametrize("function", [uproot.create, uproot.recreate, uproot.update])
def test_unknown_options_still_rejected(tmp_path, function):
    path = os.path.join(tmp_path, "sub", "file.root")
    # an fsspec option outside storage_options is not silently accepted
    with pytest.raises(TypeError, match="auto_mkdir"):
        function(path, auto_mkdir=True)
    assert not os.path.exists(os.path.join(tmp_path, "sub"))


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
    assert all(options["storage_options"] == {"auto_mkdir": True} for options in seen)
    with uproot.open(os.path.join(tmp_path, "data-part0.root")) as file:
        assert file["tree"].num_entries == 1
