# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os
import sys

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

    expected = [
        b"******",
        b"    ",
        b"...",
        b"+++++++",
        b"!!!!!",
        b"@@@@@",
    ]

    for num_workers in [1, 2]:
        with uproot.source.file.MultithreadedFileSource(
            filename, num_workers=num_workers
        ) as source:
            for i, (start, stop) in enumerate(
                [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
            ):
                chunk = source.chunk(start, stop)
                assert tobytes(chunk.raw_data) == expected[i]

        with pytest.raises(Exception):
            uproot.source.file.MultithreadedFileSource(
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

    with uproot.source.file.MemmapSource(filename, num_fallback_workers=1) as source:
        for i, (start, stop) in enumerate(
            [(0, 6), (6, 10), (10, 13), (13, 20), (20, 25), (25, 30)]
        ):
            chunk = source.chunk(start, stop)
            assert tobytes(chunk.raw_data) == expected[i]

    with pytest.raises(Exception):
        uproot.source.file.MemmapSource(
            filename + "-does-not-exist", num_fallback_workers=1
        )


@pytest.mark.skip(reason="RECHECK: example.com is flaky, too")
@pytest.mark.network
def test_http():
    for num_workers in [1, 2]:
        with uproot.source.http.MultithreadedHTTPSource(
            "https://example.com", num_workers=num_workers, timeout=10
        ) as source:
            for start, stop in [(0, 100), (50, 55), (200, 400)]:
                chunk = source.chunk(start, stop)
                assert len(tobytes(chunk.raw_data)) == stop - start


@pytest.mark.network
def test_http_fail():
    for num_workers in [1, 2]:
        with pytest.raises(Exception):
            with uproot.source.http.MultithreadedHTTPSource(
                "https://wonky.cern/does-not-exist",
                num_workers=num_workers,
                timeout=0.1,
            ) as source:
                source.chunk(0, 100)


@pytest.mark.skip(reason="RECHECK: example.com is flaky, too")
@pytest.mark.network
def test_http_multipart():
    with uproot.source.http.HTTPSource(
        "https://example.com", timeout=10, num_fallback_workers=1
    ) as source:
        for start, stop in [(0, 100), (50, 55), (200, 400)]:
            chunk = source.chunk(start, stop)
            assert len(tobytes(chunk.raw_data)) == stop - start


@pytest.mark.network
def test_http_multipart_fail():
    with pytest.raises(Exception):
        with uproot.source.http.HTTPSource(
            "https://wonky.cern/does-not-exist", timeout=0.1, num_fallback_workers=1
        ) as source:
            tobytes(source.chunk(0, 100).raw_data)


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
        timeout=10,
    ) as source:
        one = tobytes(source.chunk(0, 100).raw_data)
        assert len(one) == 100
        two = tobytes(source.chunk(50, 55).raw_data)
        assert len(two) == 5
        three = tobytes(source.chunk(200, 400).raw_data)
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.skip(
    reason="RECHECK: Run2012B_DoubleMuParked.root is super-flaky right now"
)
@pytest.mark.network
@pytest.mark.xrootd
def test_xrootd_worker():
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        num_workers=5,
        timeout=10,
    ) as source:
        one = tobytes(source.chunk(0, 100).raw_data)
        assert len(one) == 100
        two = tobytes(source.chunk(50, 55).raw_data)
        assert len(two) == 5
        three = tobytes(source.chunk(200, 400).raw_data)
        assert len(three) == 200
        assert one[:4] == b"root"


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
        one = tobytes(source.chunk(0, 100).raw_data)
        assert len(one) == 100
        two = tobytes(source.chunk(50, 55).raw_data)
        assert len(two) == 5
        three = tobytes(source.chunk(200, 400).raw_data)
        assert len(three) == 200
        assert one[:4] == b"root"
