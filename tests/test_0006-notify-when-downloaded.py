# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

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

    notifications = queue.Queue()
    with uproot4.source.file.FileSource(filename) as source:
        chunks = source.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)],
            notifications=notifications,
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


def test_file_workers(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    notifications = queue.Queue()
    with uproot4.source.file.FileSource(filename, num_workers=5) as source:
        chunks = source.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)],
            notifications=notifications,
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


def test_memmap(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    notifications = queue.Queue()
    with uproot4.source.memmap.MemmapSource(filename) as source:
        chunks = source.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)],
            notifications=notifications,
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_http_multipart():
    notifications = queue.Queue()
    with uproot4.source.http.HTTPSource("https://example.com") as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_http():
    notifications = queue.Queue()
    with uproot4.source.http.MultithreadedHTTPSource("https://example.com") as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_http_workers():
    notifications = queue.Queue()
    with uproot4.source.http.MultithreadedHTTPSource(
        "https://example.com", num_workers=2
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_http_fallback():
    notifications = queue.Queue()
    with uproot4.source.http.HTTPSource(
        "https://scikit-hep.org/uproot/examples/Zmumu.root"
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_http_fallback_workers():
    notifications = queue.Queue()
    with uproot4.source.http.HTTPSource(
        "https://scikit-hep.org/uproot/examples/Zmumu.root", num_fallback_workers=5
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_xrootd():
    pytest.importorskip("pyxrootd")
    notifications = queue.Queue()
    with uproot4.source.xrootd.MultiThreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_xrootd_workers():
    pytest.importorskip("pyxrootd")
    notifications = queue.Queue()
    with uproot4.source.xrootd.MultiThreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        num_workers=5,
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_xrootd_vectorread():
    pytest.importorskip("pyxrootd")
    notifications = queue.Queue()
    with uproot4.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = dict(((chunk.start, chunk.stop), chunk) for chunk in chunks)
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))
