# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
fsspec's cat_ranges returns exceptions in its result list by default. A read
must raise an OSError for that range, not a TypeError from slicing it.
"""

from __future__ import annotations

import fsspec
import pytest
import skhep_testdata
from fsspec.implementations.memory import MemoryFileSystem

import uproot


class PayloadError(Exception):
    """Not an OSError, like aiohttp.ClientPayloadError."""


class FailingMemoryFileSystem(MemoryFileSystem):
    protocol = "failingmemory"
    fail = False

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(path.removeprefix(f"{cls.protocol}://"))

    def cat_ranges(self, paths, starts, ends, **kwargs):
        result = super().cat_ranges(paths, starts, ends, **kwargs)
        if not self.fail:
            return result
        return [PayloadError("Response payload is not completed") for _ in result]


@pytest.fixture
def failing_file():
    fsspec.register_implementation(
        FailingMemoryFileSystem.protocol, FailingMemoryFileSystem, clobber=True
    )
    with open(skhep_testdata.data_path("uproot-Zmumu.root"), "rb") as f:
        data = f.read()
    path = "/test_1450/Zmumu.root"
    fs = FailingMemoryFileSystem()
    fs.pipe_file(path, data)
    yield f"{FailingMemoryFileSystem.protocol}://{path}"
    fs.rm(path)
    FailingMemoryFileSystem.fail = False


def test_dropped_connection_raises_oserror(failing_file):
    with uproot.open(failing_file, handler=uproot.source.fsspec.FSSpecSource) as f:
        tree = f["events"]
        FailingMemoryFileSystem.fail = True
        with pytest.raises(OSError, match="Zmumu.root") as excinfo:
            tree.arrays(["px1", "py1"], library="np")

    assert isinstance(excinfo.value.__cause__, PayloadError)


@pytest.mark.parametrize("open_files", [True, False])
@pytest.mark.parametrize("allowed", [True, (OSError,), (PayloadError,)])
def test_dropped_connection_in_dask_report(failing_file, allowed, open_files):
    dask = pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    collection, report = uproot.dask(
        {failing_file: "events"},
        library="ak",
        open_files=open_files,
        allow_read_errors_with_report=allowed,
        handler=uproot.source.fsspec.FSSpecSource,
    )
    FailingMemoryFileSystem.fail = True
    _, creport = dask.compute(collection, report, scheduler="synchronous")

    assert creport[0].exception == "OSError"
    assert "Zmumu.root" in creport[0].message
    assert "Response payload is not completed" in creport[0].message


@pytest.mark.parametrize("open_files", [True, False])
def test_dropped_connection_not_allowed_in_dask_report(failing_file, open_files):
    dask = pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    collection, report = uproot.dask(
        {failing_file: "events"},
        library="ak",
        open_files=open_files,
        allow_read_errors_with_report=(ValueError,),
        handler=uproot.source.fsspec.FSSpecSource,
    )
    FailingMemoryFileSystem.fail = True
    with pytest.raises(OSError, match="Zmumu.root"):
        dask.compute(collection, report, scheduler="synchronous")
