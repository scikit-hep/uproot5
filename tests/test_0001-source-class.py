# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os
import platform
import queue
import sys
from io import StringIO

import numpy
import pytest

import uproot


def tobytes(x):
    if hasattr(x, "tobytes"):
        return x.tobytes()
    else:
        return x.tostring()


def test_file(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    for num_workers in [1, 2]:
        source = uproot.source.file.MultithreadedFileSource(
            filename, num_workers=num_workers
        )
        with source as tmp:
            notifications = queue.Queue()
            chunks = tmp.chunks(
                [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)],
                notifications,
            )
            assert [tobytes(chunk.raw_data) for chunk in chunks] == [
                b"******",
                b"    ",
                b"...",
                b"+++++++",
                b"!!!!!",
                b"@@@@@",
            ]

        assert source.num_bytes == 30


def test_file_fail(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    for num_workers in [1, 2]:
        with pytest.raises(Exception):
            uproot.source.file.MultithreadedFileSource(
                filename + "-does-not-exist", num_workers=num_workers
            )


def test_memmap(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    source = uproot.source.file.MemmapSource(filename, num_fallback_workers=1)
    with source as tmp:
        notifications = queue.Queue()
        chunks = tmp.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)], notifications
        )
        assert [tobytes(chunk.raw_data) for chunk in chunks] == [
            b"******",
            b"    ",
            b"...",
            b"+++++++",
            b"!!!!!",
            b"@@@@@",
        ]

        assert source.num_bytes == 30


def test_memmap_fail(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with pytest.raises(Exception):
        uproot.source.file.MultithreadedFileSource(filename + "-does-not-exist")


@pytest.mark.skip(reason="RECHECK: example.com is flaky, too")
@pytest.mark.network
def test_http():
    source = uproot.source.http.HTTPSource(
        "https://example.com", timeout=10, num_fallback_workers=1
    )
    with source as tmp:
        notifications = queue.Queue()
        chunks = tmp.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
    assert source.fallback is None

    source = uproot.source.http.MultithreadedHTTPSource(
        "https://example.com", num_workers=1, timeout=10
    )
    with source as tmp:
        notifications = queue.Queue()
        chunks = tmp.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        assert [tobytes(x.raw_data) for x in chunks] == [one, two, three]


@pytest.mark.skip(reason="RECHECK: example.com is flaky, too")
def colons_and_ports():
    assert uproot._util.file_object_path_split("https://example.com:443") == (
        "https://example.com:443",
        None,
    )
    assert uproot._util.file_object_path_split("https://example.com:443/something") == (
        "https://example.com:443/something",
        None,
    )
    assert uproot._util.file_object_path_split(
        "https://example.com:443/something:else"
    ) == ("https://example.com:443/something", "else")


@pytest.mark.skip(reason="RECHECK: example.com is flaky, too")
@pytest.mark.network
def test_http_port():
    source = uproot.source.http.HTTPSource(
        "https://example.com:443", timeout=10, num_fallback_workers=1
    )
    with source as tmp:
        notifications = queue.Queue()
        chunks = tmp.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200

    source = uproot.source.http.MultithreadedHTTPSource(
        "https://example.com:443", num_workers=1, timeout=10
    )
    with source as tmp:
        notifications = queue.Queue()
        chunks = tmp.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        assert [tobytes(x.raw_data) for x in chunks] == [one, two, three]


@pytest.mark.network
def test_http_size():
    with uproot.source.http.HTTPSource(
        "https://scikit-hep.org/uproot3/examples/Zmumu.root",
        timeout=10,
        num_fallback_workers=1,
    ) as source:
        size1 = source.num_bytes

    with uproot.source.http.MultithreadedHTTPSource(
        "https://scikit-hep.org/uproot3/examples/Zmumu.root", num_workers=1, timeout=10
    ) as source:
        size2 = source.num_bytes

    assert size1 == size2


@pytest.mark.network
def test_http_size_port():
    with uproot.source.http.HTTPSource(
        "https://scikit-hep.org:443/uproot3/examples/Zmumu.root",
        timeout=10,
        num_fallback_workers=1,
    ) as source:
        size1 = source.num_bytes

    with uproot.source.http.MultithreadedHTTPSource(
        "https://scikit-hep.org:443/uproot3/examples/Zmumu.root",
        num_workers=1,
        timeout=10,
    ) as source:
        size2 = source.num_bytes

    assert size1 == size2


@pytest.mark.network
def test_http_fail():
    source = uproot.source.http.HTTPSource(
        "https://wonky.cern/does-not-exist", timeout=0.1, num_fallback_workers=1
    )
    with pytest.raises(Exception) as err:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        chunks[0].raw_data


@pytest.mark.network
def test_no_multipart():
    for num_workers in [1, 2]:
        with uproot.source.http.MultithreadedHTTPSource(
            "https://scikit-hep.org/uproot3/examples/Zmumu.root",
            num_workers=num_workers,
            timeout=10,
        ) as source:
            notifications = queue.Queue()
            chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
            one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
            assert len(one) == 100
            assert len(two) == 5
            assert len(three) == 200
            assert one[:4] == b"root"


@pytest.mark.network
def test_no_multipart_fail():
    for num_workers in [1, 2]:
        source = uproot.source.http.MultithreadedHTTPSource(
            "https://wonky.cern/does-not-exist", num_workers=num_workers, timeout=0.1
        )
        with pytest.raises(Exception) as err:
            notifications = queue.Queue()
            chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
            chunks[0].raw_data


@pytest.mark.network
def test_fallback():
    for num_workers in [1, 2]:
        with uproot.source.http.HTTPSource(
            "https://scikit-hep.org/uproot3/examples/Zmumu.root",
            timeout=10,
            num_fallback_workers=num_workers,
        ) as source:
            notifications = queue.Queue()
            chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
            one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
            assert len(one) == 100
            assert len(two) == 5
            assert len(three) == 200
            assert one[:4] == b"root"


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd():
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        num_workers=1,
        timeout=20,
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_deadlock():
    pytest.importorskip("XRootD")
    # Attach this file to the "test_xrootd_deadlock" function so it leaks
    pytest.uproot_test_xrootd_deadlock_f = uproot.source.xrootd.XRootDResource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=20,
    )


@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_fail():
    pytest.importorskip("XRootD")
    with pytest.raises(Exception) as err:
        source = uproot.source.xrootd.MultithreadedXRootDSource(
            "root://wonky.cern/does-not-exist", num_workers=1, timeout=1
        )


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_vectorread():
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_vectorread_max_element_split():
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
    ) as source:
        notifications = queue.Queue()
        max_element_size = 2097136
        chunks = source.chunks([(0, max_element_size + 1)], notifications)
        (one,) = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == max_element_size + 1


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_vectorread_max_element_split_consistency():
    pytest.importorskip("XRootD")
    filename = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"

    def get_chunk(Source, **kwargs):
        with Source(filename, **kwargs) as source:
            notifications = queue.Queue()
            max_element_size = 2097136
            chunks = source.chunks([(0, max_element_size + 1)], notifications)
            (one,) = (tobytes(chunk.raw_data) for chunk in chunks)
            return one

    chunk1 = get_chunk(
        uproot.source.xrootd.XRootDSource,
        timeout=10,
        max_num_elements=None,
        num_workers=1,
    )
    chunk2 = get_chunk(
        uproot.source.xrootd.MultithreadedXRootDSource, timeout=10, num_workers=1
    )
    assert chunk1 == chunk2


@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_vectorread_fail():
    pytest.importorskip("XRootD")
    with pytest.raises(Exception) as err:
        source = uproot.source.xrootd.XRootDSource(
            "root://wonky.cern/does-not-exist",
            timeout=1,
            max_num_elements=None,
            num_workers=1,
        )


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_size():
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
    ) as source:
        size1 = source.num_bytes

    pytest.importorskip("XRootD")
    with uproot.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        num_workers=1,
    ) as source:
        size2 = source.num_bytes

    assert size1 == size2
    assert size1 == 3469136394


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_numpy_int():
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
    ) as source:
        chunk = source.chunk(numpy.int64(0), numpy.int64(100))
        assert len(chunk.raw_data) == 100


def test_cursor_debug():
    data = numpy.concatenate(
        [
            numpy.array([123, 123, 123], "u1"),
            numpy.array(
                [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 101, 202, 303], ">f4"
            ).view("u1"),
            numpy.array([123, 123], "u1"),
        ]
    )
    future = uproot.source.futures.TrivialFuture(data)

    chunk = uproot.source.chunk.Chunk(None, 0, len(data), future)
    cursor = uproot.source.cursor.Cursor(0)

    output = StringIO()
    cursor.debug(chunk, offset=3, dtype=">f4", stream=output)
    assert (
        output.getvalue()
        == """--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
  {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
                        1.1             2.2             3.3             4.4
    --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
    176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
    --- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
            5.5             6.6             7.7             8.8             9.9
    --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
    202   0   0  67  74   0   0  67 151 128   0 123 123
    --- --- ---   C   J --- ---   C --- --- ---   {   {
          101.0           202.0           303.0
"""
    )

    output = StringIO()
    cursor.debug(chunk, stream=output)
    assert (
        output.getvalue()
        == """--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
  {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
--- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
--+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
202   0   0  67  74   0   0  67 151 128   0 123 123
--- --- ---   C   J --- ---   C --- --- ---   {   {
"""
    )
