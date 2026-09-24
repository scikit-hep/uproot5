# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for issue #1688: ``uproot.recreate`` must truncate.

Before the move to fsspec, ``uproot.recreate`` opened the path with mode ``"w"``,
which truncated it. Afterwards it went through ``FileSink``, which only truncates
when the file does *not* already exist, so recreating over a larger file left
every byte beyond the new ``fEND`` in place.
"""

from __future__ import annotations

import io

import fsspec
import pytest

import uproot


@pytest.mark.parametrize("kind", ["str", "pathlib", "file-like"])
def test_recreate_truncates(tmp_path, kind):
    path = tmp_path / "file.root"
    path.write_bytes(b"\x00" * 100_000)

    if kind == "file-like":
        with open(path, "r+b") as file, uproot.recreate(file) as f:
            f["h"] = "hello"
    else:
        with uproot.recreate(str(path) if kind == "str" else path) as f:
            f["h"] = "hello"

    with uproot.open(path) as f:
        assert f.keys() == ["h;1"]
        assert path.stat().st_size == f.file.fEND


def test_recreate_file_like_without_truncate():
    class FileLike:
        def __init__(self):
            self.buffer = io.BytesIO()

        def __getattr__(self, name):
            if name in ("read", "write", "seek", "tell", "flush"):
                return getattr(self.buffer, name)
            raise AttributeError(name)

    file = FileLike()
    with uproot.recreate(file) as f:
        f["h"] = "hello"
    with uproot.open(io.BytesIO(file.buffer.getvalue())) as f:
        assert f.keys() == ["h;1"]


def test_recreate_keeps_file_if_it_cannot_be_opened(monkeypatch):
    # like S3 or XRootD, which can write whole files but not open them "r+b"
    fs = fsspec.filesystem("memory")
    original_open = type(fs)._open

    def _open(self, path, mode="rb", **kwargs):
        if mode == "r+b":
            raise NotImplementedError("File mode not supported")
        return original_open(self, path, mode, **kwargs)

    monkeypatch.setattr(type(fs), "_open", _open)
    fs.pipe("/test_1697/file.root", b"contents")
    try:
        with pytest.raises(NotImplementedError):
            uproot.recreate("memory://test_1697/file.root")
        assert fs.cat("/test_1697/file.root") == b"contents"
    finally:
        fs.rm("/test_1697", recursive=True)


def test_create_and_update_do_not_truncate(tmp_path):
    path = tmp_path / "subdir" / "file.root"
    with uproot.create(path) as f:
        f["h"] = "hello"
    contents = path.read_bytes()

    with pytest.raises(FileExistsError):
        uproot.create(path)
    assert path.read_bytes() == contents

    # a file-like object has no path to check, so create writes into it as is
    other = tmp_path / "other.root"
    other.write_bytes(b"\x00" * 100_000)
    with open(other, "r+b") as file, uproot.create(file) as f:
        f["h"] = "hello"
    assert other.stat().st_size == 100_000

    with uproot.update(path) as f:
        f["h2"] = "world"
    with uproot.open(path) as f:
        assert f.keys() == ["h;1", "h2;1"]


@pytest.mark.parametrize("function", [uproot.create, uproot.recreate, uproot.update])
def test_invalid_option_leaves_file_intact(tmp_path, function):
    path = tmp_path / "file.root"
    with uproot.recreate(path) as f:
        f["h"] = "hello"
    contents = path.read_bytes()

    with pytest.raises(TypeError, match="compresion"):
        function(path, compresion=uproot.ZLIB(1))
    assert path.read_bytes() == contents
