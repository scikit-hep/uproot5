# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO
try:
    import queue
except ImportError:
    import Queue as queue

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

    expected = [
        b"******",
        b"    ",
        b"...",
        b"+++++++",
        b"!!!!!",
        b"@@@@@",
    ]

    for num_workers in [0, 1, 2]:
        with uproot4.source.file.FileSource(
            filename, num_workers=num_workers
        ) as source:
            for i, (start, stop) in enumerate(
                [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
            ):
                chunk = source.chunk(start, stop)
                assert chunk.raw_data.tostring() == expected[i]

        with pytest.raises(Exception):
            uproot4.source.file.FileSource(
                filename + "-does-not-exist", num_workers=num_workers
            )


def test_memmap(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    expected = [
        b"******",
        b"    ",
        b"...",
        b"+++++++",
        b"!!!!!",
        b"@@@@@",
    ]

    with uproot4.source.memmap.MemmapSource(filename) as source:
        for i, (start, stop) in enumerate(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
        ):
            chunk = source.chunk(start, stop)
            assert chunk.raw_data.tostring() == expected[i]

    with pytest.raises(Exception):
        uproot4.source.memmap.MemmapSource(filename + "-does-not-exist")


def test_http():
    for num_workers in [0, 1, 2]:
        with uproot4.source.http.HTTPSource(
            "https://example.com", num_workers=num_workers
        ) as source:
            for start, stop in [(0, 100), (50, 55), (200, 400)]:
                chunk = source.chunk(start, stop)
                assert len(chunk.raw_data.tostring()) == stop - start

            with pytest.raises(Exception):
                with uproot4.source.http.HTTPSource(
                    "https://wonky.cern/does-not-exist", num_workers=num_workers
                ) as source:
                    source.chunk(0, 100)


def test_http_multipart():
    with uproot4.source.http.HTTPMultipartSource("https://example.com") as source:
        for start, stop in [(0, 100), (50, 55), (200, 400)]:
            chunk = source.chunk(start, stop)
            assert len(chunk.raw_data.tostring()) == stop - start

        with pytest.raises(Exception):
            with uproot4.source.http.HTTPMultipartSource(
                "https://wonky.cern/does-not-exist"
            ) as source:
                source.chunk(0, 100).raw_data.tostring()


def test_xrootd():
    pytest.importorskip("pyxrootd")
    with uproot4.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"
    ) as source:
        one = source.chunk(0, 100).raw_data.tostring()
        assert len(one) == 100
        two = source.chunk(50, 55).raw_data.tostring()
        assert len(two) == 5
        three = source.chunk(200, 400).raw_data.tostring()
        assert len(three) == 200
        assert one[:4] == b"root"


def test_xrootd_worker():
    pytest.importorskip("pyxrootd")
    with uproot4.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        num_workers=5,
    ) as source:
        one = source.chunk(0, 100).raw_data.tostring()
        assert len(one) == 100
        two = source.chunk(50, 55).raw_data.tostring()
        assert len(two) == 5
        three = source.chunk(200, 400).raw_data.tostring()
        assert len(three) == 200
        assert one[:4] == b"root"


def test_xrootd_vectorread():
    pytest.importorskip("pyxrootd")
    with uproot4.source.xrootd.XRootDVectorReadSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"
    ) as source:
        one = source.chunk(0, 100).raw_data.tostring()
        assert len(one) == 100
        two = source.chunk(50, 55).raw_data.tostring()
        assert len(two) == 5
        three = source.chunk(200, 400).raw_data.tostring()
        assert len(three) == 200
        assert one[:4] == b"root"
