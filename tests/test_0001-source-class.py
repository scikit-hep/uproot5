# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

import numpy
import pytest

import uproot4
import uproot4.source.futures
import uproot4.source.cursor
import uproot4.source.chunk
import uproot4.source.file
import uproot4.source.memmap
import uproot4.source.http
import uproot4.source.xrootd


def test_file(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    for num_workers in [0, 1, 2]:
        source = uproot4.source.file.FileSource(filename, num_workers=num_workers)
        with source as tmp:
            chunks = tmp.chunks(
                [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
            )
            assert [chunk.raw_data.tostring() for chunk in chunks] == [
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

    for num_workers in [0, 1, 2]:
        with pytest.raises(Exception):
            uproot4.source.file.FileSource(
                filename + "-does-not-exist", num_workers=num_workers
            )


def test_memmap(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    source = uproot4.source.memmap.MemmapSource(filename, num_fallback_workers=0)
    with source as tmp:
        chunks = tmp.chunks([(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)])
        assert [chunk.raw_data.tostring() for chunk in chunks] == [
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
        uproot4.source.file.FileSource(filename + "-does-not-exist")


@pytest.mark.network
def test_http():
    source = uproot4.source.http.HTTPSource(
        "https://example.com", timeout=10, num_fallback_workers=0
    )
    with source as tmp:
        chunks = tmp.chunks([(0, 100), (50, 55), (200, 400)])
        one, two, three = [chunk.raw_data.tostring() for chunk in chunks]
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200

    source = uproot4.source.http.MultithreadedHTTPSource(
        "https://example.com", num_workers=0, timeout=10
    )
    with source as tmp:
        chunks = tmp.chunks([(0, 100), (50, 55), (200, 400)])
        assert [x.raw_data.tostring() for x in chunks] == [one, two, three]


@pytest.mark.network
def test_http_size():
    with uproot4.source.http.HTTPSource(
        "https://scikit-hep.org/uproot/examples/Zmumu.root",
        timeout=10,
        num_fallback_workers=0,
    ) as source:
        size1 = source.num_bytes

    with uproot4.source.http.MultithreadedHTTPSource(
        "https://scikit-hep.org/uproot/examples/Zmumu.root", num_workers=0, timeout=10
    ) as source:
        size2 = source.num_bytes

    assert size1 == size2


@pytest.mark.network
def test_http_fail():
    source = uproot4.source.http.HTTPSource(
        "https://wonky.cern/does-not-exist", timeout=0.1, num_fallback_workers=0
    )
    with pytest.raises(Exception) as err:
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)])
        chunks[0].raw_data


@pytest.mark.network
def test_no_multipart():
    for num_workers in [0, 1, 2]:
        with uproot4.source.http.MultithreadedHTTPSource(
            "https://scikit-hep.org/uproot/examples/Zmumu.root",
            num_workers=num_workers,
            timeout=10,
        ) as source:
            chunks = source.chunks([(0, 100), (50, 55), (200, 400)])
            one, two, three = [chunk.raw_data.tostring() for chunk in chunks]
            assert len(one) == 100
            assert len(two) == 5
            assert len(three) == 200
            assert one[:4] == b"root"


@pytest.mark.network
def test_no_multipart_fail():
    for num_workers in [0, 1, 2]:
        source = uproot4.source.http.MultithreadedHTTPSource(
            "https://wonky.cern/does-not-exist", num_workers=num_workers, timeout=0.1
        )
        with pytest.raises(Exception) as err:
            chunks = source.chunks([(0, 100), (50, 55), (200, 400)])
            chunks[0].raw_data


@pytest.mark.network
def test_fallback():
    for num_workers in [0, 1, 2]:
        with uproot4.source.http.HTTPSource(
            "https://scikit-hep.org/uproot/examples/Zmumu.root",
            timeout=10,
            num_fallback_workers=num_workers,
        ) as source:
            chunks = source.chunks([(0, 100), (50, 55), (200, 400)])
            one, two, three = [chunk.raw_data.tostring() for chunk in chunks]
            assert len(one) == 100
            assert len(two) == 5
            assert len(three) == 200
            assert one[:4] == b"root"


@pytest.mark.network
@pytest.mark.skip(reason="Cannot open the file for some reason.")
def test_xrootd():
    pytest.importorskip("pyxrootd")
    with uproot4.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        num_workers=0,
        timeout=20,
    ) as source:
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)])
        one, two, three = [chunk.raw_data.tostring() for chunk in chunks]
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.network
def test_xrootd_fail():
    pytest.importorskip("pyxrootd")
    with pytest.raises(Exception) as err:
        source = uproot4.source.xrootd.MultithreadedXRootDSource(
            "root://wonky.cern/does-not-exist", num_workers=0, timeout=1
        )


@pytest.mark.network
def test_xrootd_vectorread():
    pytest.importorskip("pyxrootd")
    with uproot4.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
    ) as source:
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)])
        one, two, three = [chunk.raw_data.tostring() for chunk in chunks]
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.network
def test_xrootd_vectorread_fail():
    pytest.importorskip("pyxrootd")
    with pytest.raises(Exception) as err:
        source = uproot4.source.xrootd.XRootDSource(
            "root://wonky.cern/does-not-exist", timeout=1, max_num_elements=None
        )


@pytest.mark.network
def test_xrootd_size():
    pytest.importorskip("pyxrootd")
    with uproot4.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
    ) as source:
        size1 = source.num_bytes

    pytest.importorskip("pyxrootd")
    with uproot4.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        num_workers=0,
    ) as source:
        size2 = source.num_bytes

    assert size1 == size2
    assert size1 == 3469136394


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
    future = uproot4.source.futures.TrivialFuture(data)

    chunk = uproot4.source.chunk.Chunk(None, 0, len(data), future, True)
    cursor = uproot4.source.cursor.Cursor(0)

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
