# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os
import queue
import sys
from io import StringIO

import numpy
import pytest

import uproot


def test_file(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    notifications = queue.Queue()
    with uproot.source.file.MultithreadedFileSource(filename, num_workers=1) as source:
        chunks = source.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)],
            notifications=notifications,
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


def test_file_workers(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    notifications = queue.Queue()
    with uproot.source.file.MultithreadedFileSource(filename, num_workers=5) as source:
        chunks = source.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)],
            notifications=notifications,
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


def test_memmap(tmpdir):
    filename = os.path.join(str(tmpdir), "tmp.raw")
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    notifications = queue.Queue()
    with uproot.source.file.MemmapSource(filename, num_fallback_workers=1) as source:
        chunks = source.chunks(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)],
            notifications=notifications,
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.skip(reason="RECHECK: example.com is flaky, too")
@pytest.mark.network
def test_http_multipart():
    notifications = queue.Queue()
    with uproot.source.http.HTTPSource(
        "https://example.com", timeout=10, num_fallback_workers=1
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.skip(reason="RECHECK: example.com is flaky, too")
@pytest.mark.network
def test_http():
    notifications = queue.Queue()
    with uproot.source.http.MultithreadedHTTPSource(
        "https://example.com", timeout=10, num_workers=1
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.skip(reason="RECHECK: example.com is flaky, too")
@pytest.mark.network
def test_http_workers():
    notifications = queue.Queue()
    with uproot.source.http.MultithreadedHTTPSource(
        "https://example.com", timeout=10, num_workers=2
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_http_fallback():
    notifications = queue.Queue()
    with uproot.source.http.HTTPSource(
        "https://scikit-hep.org/uproot3/examples/Zmumu.root",
        timeout=10,
        num_fallback_workers=1,
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.network
def test_http_fallback_workers():
    notifications = queue.Queue()
    with uproot.source.http.HTTPSource(
        "https://scikit-hep.org/uproot3/examples/Zmumu.root",
        timeout=10,
        num_fallback_workers=5,
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd():
    pytest.importorskip("XRootD")
    notifications = queue.Queue()
    with uproot.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        num_workers=1,
        timeout=10,
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_workers():
    pytest.importorskip("XRootD")
    notifications = queue.Queue()
    with uproot.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        num_workers=5,
        timeout=10,
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_vectorread():
    pytest.importorskip("XRootD")
    notifications = queue.Queue()
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
    ) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))
