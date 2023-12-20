# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import queue
from io import StringIO

import numpy
import pytest
import threading
import uproot


def tobytes(x):
    if hasattr(x, "tobytes"):
        return x.tobytes()
    else:
        return x.tostring()


@pytest.fixture
def use_threads(request):
    if request.param:
        yield
        return
    else:
        print("CHECK")
        n_threads = threading.active_count()
        yield request.param
        assert threading.active_count() == n_threads


@pytest.mark.parametrize(
    "use_threads, num_workers",
    [(True, 1), (True, 2), (False, 0)],
)
def test_file(use_threads, num_workers, tmp_path):
    filename = tmp_path / "tmp.raw"
    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with uproot.source.file.MultithreadedFileSource(
        filename, num_workers=num_workers, use_threads=use_threads
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks(
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


@pytest.mark.parametrize(
    "use_threads, num_workers",
    [(True, 1), (True, 2), (False, 0)],
)
def test_file_fail(use_threads, num_workers, tmp_path):
    filename = tmp_path / "tmp.raw"

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with pytest.raises(Exception):
        uproot.source.file.MultithreadedFileSource(
            filename + "-does-not-exist",
            num_workers=num_workers,
            use_threads=use_threads,
        )


@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_memmap(use_threads, tmp_path):
    filename = tmp_path / "tmp.raw"

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with uproot.source.file.MemmapSource(
        filename, num_fallback_workers=1, use_threads=use_threads
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks(
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


@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_memmap_fail(use_threads, tmp_path):
    filename = tmp_path / "tmp.raw"

    with open(filename, "wb") as tmp:
        tmp.write(b"******    ...+++++++!!!!!@@@@@")

    with pytest.raises(Exception):
        with uproot.source.file.MemmapSource(
            tmp_path / f"{filename.name}-does-not-exist",
            num_fallback_workers=1,
            use_threads=use_threads,
        ):
            ...


@pytest.mark.parametrize("use_threads", [True, False])
@pytest.mark.network
def test_http(use_threads):
    url = "https://example.com"
    with uproot.source.http.HTTPSource(
        url,
        timeout=10,
        num_fallback_workers=1,
        use_threads=use_threads,
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert source.fallback is None

    with uproot.source.http.MultithreadedHTTPSource(
        url, num_workers=1, timeout=10, use_threads=use_threads
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        assert [tobytes(chunk.raw_data) for chunk in chunks] == [one, two, three]


def test_colons_and_ports():
    assert uproot._util.file_object_path_split("https://example.com:443") == (
        "https://example.com:443",
        None,
    )
    assert uproot._util.file_object_path_split("https://example.com:443/file.root") == (
        "https://example.com:443/file.root",
        None,
    )
    assert uproot._util.file_object_path_split(
        "https://example.com:443/file.root:object"
    ) == ("https://example.com:443/file.root", "object")


@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
@pytest.mark.network
def test_http_port(use_threads):
    source = uproot.source.http.HTTPSource(
        "https://example.com:443",
        timeout=10,
        num_fallback_workers=1,
        use_threads=use_threads,
    )
    with source as tmp:
        notifications = queue.Queue()
        chunks = tmp.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200

    source = uproot.source.http.MultithreadedHTTPSource(
        "https://example.com:443", num_workers=1, timeout=10, use_threads=use_threads
    )
    with source as tmp:
        notifications = queue.Queue()
        chunks = tmp.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        assert [tobytes(x.raw_data) for x in chunks] == [one, two, three]


@pytest.mark.parametrize("use_threads", [True, False])
def test_http_size(http_server, use_threads):
    url = f"{http_server}/uproot-issue121.root"
    with uproot.source.http.HTTPSource(
        url,
        timeout=10,
        num_fallback_workers=1,
        use_threads=use_threads,
    ) as source:
        size1 = source.num_bytes

    with uproot.source.http.MultithreadedHTTPSource(
        url,
        num_workers=1,
        timeout=10,
        use_threads=use_threads,
    ) as source:
        size2 = source.num_bytes

    assert size1 == size2


@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
@pytest.mark.network
def test_http_size_port(use_threads):
    with uproot.source.http.HTTPSource(
        "https://scikit-hep.org:443/uproot3/examples/Zmumu.root",
        timeout=10,
        num_fallback_workers=1,
        use_threads=use_threads,
    ) as source:
        size1 = source.num_bytes

    with uproot.source.http.MultithreadedHTTPSource(
        "https://scikit-hep.org:443/uproot3/examples/Zmumu.root",
        num_workers=1,
        timeout=10,
        use_threads=use_threads,
    ) as source:
        size2 = source.num_bytes

    assert size1 == size2


@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
@pytest.mark.network
def test_http_fail(use_threads):
    source = uproot.source.http.HTTPSource(
        "https://wonky.cern/does-not-exist",
        timeout=0.1,
        num_fallback_workers=1,
        use_threads=use_threads,
    )
    with pytest.raises(Exception):
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        chunks[0].raw_data


@pytest.mark.parametrize(
    "use_threads, num_workers",
    [(True, 1), (True, 2), (False, 0)],
)
@pytest.mark.network
def test_no_multipart(use_threads, num_workers):
    with uproot.source.http.MultithreadedHTTPSource(
        "https://scikit-hep.org/uproot3/examples/Zmumu.root",
        num_workers=num_workers,
        timeout=10,
        use_threads=use_threads,
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.parametrize(
    "use_threads, num_workers",
    [(True, 1), (True, 2), (False, 0)],
)
@pytest.mark.network
def test_no_multipart_fail(use_threads, num_workers):
    source = uproot.source.http.MultithreadedHTTPSource(
        "https://wonky.cern/does-not-exist",
        num_workers=num_workers,
        timeout=0.1,
        use_threads=use_threads,
    )
    with pytest.raises(Exception):
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        chunks[0].raw_data


@pytest.mark.parametrize("use_threads, num_workers", [(True, 1), (True, 2), (False, 0)])
def test_fallback(http_server, use_threads, num_workers):
    url = f"{http_server}/uproot-issue121.root"
    with uproot.source.http.HTTPSource(
        url,
        timeout=10,
        num_fallback_workers=num_workers,
        use_threads=use_threads,
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd(use_threads):
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        num_workers=1,
        timeout=20,
        use_threads=use_threads,
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd_fail(use_threads):
    pytest.importorskip("XRootD")
    with pytest.raises(Exception):
        uproot.source.xrootd.MultithreadedXRootDSource(
            "root://wonky.cern/does-not-exist",
            num_workers=1,
            timeout=1,
            use_threads=use_threads,
        )


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd_vectorread(use_threads):
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
        use_threads=use_threads,
    ) as source:
        notifications = queue.Queue()
        chunks = source.chunks([(0, 100), (50, 55), (200, 400)], notifications)
        one, two, three = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == 100
        assert len(two) == 5
        assert len(three) == 200
        assert one[:4] == b"root"


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd_vectorread_max_element_split(use_threads):
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
        use_threads=use_threads,
    ) as source:
        notifications = queue.Queue()
        max_element_size = 2097136
        chunks = source.chunks([(0, max_element_size + 1)], notifications)
        (one,) = (tobytes(chunk.raw_data) for chunk in chunks)
        assert len(one) == max_element_size + 1


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd_vectorread_max_element_split_consistency(use_threads):
    pytest.importorskip("XRootD")
    filename = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"

    def get_chunk(source_cls, **kwargs):
        with source_cls(filename, **kwargs) as source:
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
        use_threads=use_threads,
    )
    chunk2 = get_chunk(
        uproot.source.xrootd.MultithreadedXRootDSource, timeout=10, num_workers=1
    )
    assert chunk1 == chunk2


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd_vectorread_fail(use_threads):
    pytest.importorskip("XRootD")
    with pytest.raises(Exception):
        uproot.source.xrootd.XRootDSource(
            "root://wonky.cern/does-not-exist",
            timeout=1,
            max_num_elements=None,
            num_workers=1,
            use_threads=use_threads,
        )


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd_size(use_threads):
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
        use_threads=use_threads,
    ) as source:
        size1 = source.num_bytes

    with uproot.source.xrootd.MultithreadedXRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        num_workers=1,
        use_threads=use_threads,
    ) as source:
        size2 = source.num_bytes

    assert size1 == size2
    assert size1 == 3469136394


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize("use_threads", [True, False], indirect=True)
def test_xrootd_numpy_int(use_threads):
    pytest.importorskip("XRootD")
    with uproot.source.xrootd.XRootDSource(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        timeout=10,
        max_num_elements=None,
        num_workers=1,
        use_threads=use_threads,
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
