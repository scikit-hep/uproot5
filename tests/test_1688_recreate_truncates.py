# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for issue #1688: ``uproot.recreate`` must truncate.

Before the move to fsspec, ``uproot.recreate`` opened the path with mode ``"w"``,
which truncated it. Afterwards it went through ``FileSink``, which only truncates
when the file does *not* already exist, so recreating over a larger file left
every byte beyond the new ``fEND`` in place.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import uproot


def test_recreate_truncates_existing_file(tmp_path):
    path = str(tmp_path / "file.root")
    with open(path, "wb") as f:
        f.write(b"\x00" * 100_000)

    with uproot.recreate(path):
        pass

    with uproot.open(path) as f:
        end = f.file.fEND

    assert os.path.getsize(path) == end
    assert os.path.getsize(path) < 100_000


def test_recreate_truncates_larger_root_file(tmp_path):
    path = str(tmp_path / "file.root")
    with uproot.recreate(path) as f:
        f["big"] = np.histogram(np.random.normal(size=100_000), bins=5_000)
    big_size = os.path.getsize(path)

    with uproot.recreate(path) as f:
        f["small"] = "hello"
    small_size = os.path.getsize(path)

    assert small_size < big_size
    with uproot.open(path) as f:
        assert f.keys() == ["small;1"]
        assert os.path.getsize(path) == f.file.fEND


def test_recreate_still_creates_missing_file_and_parents(tmp_path):
    path = str(tmp_path / "subdir" / "file.root")
    with uproot.recreate(path) as f:
        f["h"] = np.histogram(np.random.normal(size=100))
    with uproot.open(path) as f:
        assert f.keys() == ["h;1"]


def test_create_still_refuses_to_overwrite(tmp_path):
    path = str(tmp_path / "file.root")
    with uproot.create(path) as f:
        f["h"] = np.histogram(np.random.normal(size=100))
    size = os.path.getsize(path)

    with pytest.raises(FileExistsError):
        uproot.create(path)

    # the failed create must not have touched the file
    assert os.path.getsize(path) == size
    with uproot.open(path) as f:
        assert f.keys() == ["h;1"]


def test_update_does_not_truncate(tmp_path):
    path = str(tmp_path / "file.root")
    with uproot.recreate(path) as f:
        f["h"] = np.histogram(np.random.normal(size=100))

    with uproot.update(path) as f:
        f["h2"] = np.histogram(np.random.normal(size=100))

    with uproot.open(path) as f:
        assert f.keys() == ["h;1", "h2;1"]
