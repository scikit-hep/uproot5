# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest
import uproot
import uproot.source.fsspec

import queue

import skhep_testdata


@pytest.mark.network
def test_open_fsspec_http():
    with uproot.open(
        "https://github.com/scikit-hep/scikit-hep-testdata/raw/v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
        http_handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
def test_open_fsspec_github():
    pytest.skip(
        "skipping due to GitHub API rate limitations - this should work fine - see https://github.com/scikit-hep/uproot5/pull/973 for details"
    )
    with uproot.open(
        "github://scikit-hep:scikit-hep-testdata@v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
        http_handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


def test_open_fsspec_local(tmp_path):
    local_path = skhep_testdata.data_path("uproot-issue121.root")

    with uproot.open(
        local_path,
        file_handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
def test_open_fsspec_s3():
    pytest.importorskip("s3fs")

    with uproot.open(
        "s3://pivarski-princeton/pythia_ppZee_run17emb.picoDst.root:PicoDst",
        anon=True,
        s3_handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Event/Event.mEventId"].array(library="np")
        assert len(data) == 8004


@pytest.mark.network
@pytest.mark.xrootd
def test_open_fsspec_xrootd():
    pytest.importorskip("XRootD")
    with uproot.open(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        xrootd_handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/run"].array(library="np", entry_stop=20)
        assert len(data) == 20
        assert (data == 194778).all()


@pytest.mark.network
def test_fsspec_chunks():
    # Use the local HTTP server to serve test.root
    url = "https://github.com/scikit-hep/scikit-hep-testdata/raw/v0.4.33/src/skhep_testdata/data/uproot-issue121.root"

    notifications = queue.Queue()
    with uproot.source.fsspec.FSSpecSource(url) as source:
        import time

        start_time = time.time()
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        time_to_get_chunks = time.time() - start_time
        # TODO: this might not be the best way to test it is non-blocking
        # This is non-blocking, so it should be fast (0.0005390644073486328 seconds last time I ran it)
        assert (
            time_to_get_chunks < 0.01
        ), f"This should be fast (non-blocking). Elapsed: {time_to_get_chunks} seconds"
        print(f"Time to get chunks: {time_to_get_chunks} seconds")

        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))

        chunk_data_sum = {sum(chunk.raw_data) for chunk in chunks}
        assert chunk_data_sum == {3967, 413, 10985}, "Chunk data does not match"
