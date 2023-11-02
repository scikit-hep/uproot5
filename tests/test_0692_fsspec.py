# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import fsspec
import pytest
import uproot
import uproot.source.fsspec

import skhep_testdata
import queue
import subprocess


def test_open_fsspec_http(server):
    pytest.importorskip("aiohttp")

    url = f"{server}/uproot-issue121.root"
    with uproot.open(
        url,
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
@pytest.mark.skip(
    reason="skipping due to GitHub API rate limitations - this should work fine - see https://github.com/scikit-hep/uproot5/pull/973 for details"
)
def test_open_fsspec_github():
    with uproot.open(
        "github://scikit-hep:scikit-hep-testdata@v0.4.33/src/skhep_testdata/data/uproot-issue121.root",
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


def test_open_fsspec_local():
    local_path = skhep_testdata.data_path("uproot-issue121.root")

    with uproot.open(
        local_path,
        handler=uproot.source.fsspec.FSSpecSource,
    ) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
@pytest.mark.parametrize(
    "handler",
    [
        # uproot.source.fsspec.FSSpecSource,
        uproot.source.s3.S3Source,
        None,
    ],
)
def test_open_fsspec_s3(handler):
    pytest.importorskip("s3fs")

    with uproot.open(
        "s3://pivarski-princeton/pythia_ppZee_run17emb.picoDst.root:PicoDst",
        anon=True,
        handler=handler,
    ) as f:
        data = f["Event/Event.mEventId"].array(library="np")
        assert len(data) == 8004


def test_open_fsspec_ssh():
    pytest.importorskip("sshfs")

    # check localhost has ssh access to itself
    try:
        user = subprocess.check_output(["whoami"]).strip().decode("ascii")
        host = "localhost"
        ssh_command = ["ssh", f"{user}@{host}", "'echo hello'"]
        result = subprocess.run(
            ssh_command,
            shell=True,
            text=True,
            capture_output=True,
        )
        assert (
            result.returncode == 0
        ), f"ssh access to localhost failed with {result.stderr}"
    except Exception as e:
        pytest.skip(f"ssh access to localhost failed with {e}")

    # at this time sshfs does not implement cat_file. This will alert us if it ever does
    with pytest.raises(NotImplementedError):
        fs = fsspec.filesystem("ssh", host="localhost")
        fs.cat_file("some-file", start=0, end=100)

    pytest.skip("sshfs does not implement cat_file")

    # cache the file
    local_path = skhep_testdata.data_path("uproot-issue121.root")

    uri = f"ssh://{user}@{host}:22{local_path}"
    with uproot.open(uri, handler=uproot.source.fsspec.FSSpecSource) as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


@pytest.mark.network
@pytest.mark.xrootd
@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        uproot.source.xrootd.XRootDSource,
        None,
    ],
)
def test_open_fsspec_xrootd(handler):
    pytest.importorskip("XRootD")
    with uproot.open(
        "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
        handler=handler,
    ) as f:
        data = f["Events/run"].array(library="np", entry_stop=20)
        assert len(data) == 20
        assert (data == 194778).all()


def test_fsspec_chunks(server):
    pytest.importorskip("aiohttp")

    url = f"{server}/uproot-issue121.root"

    notifications = queue.Queue()
    with uproot.source.fsspec.FSSpecSource(url) as source:
        chunks = source.chunks(
            [(0, 100), (50, 55), (200, 400)], notifications=notifications
        )
        expected = {(chunk.start, chunk.stop): chunk for chunk in chunks}
        while len(expected) > 0:
            chunk = notifications.get()
            expected.pop((chunk.start, chunk.stop))

        chunk_data_sum = {sum(chunk.raw_data) for chunk in chunks}
        assert chunk_data_sum == {3967, 413, 10985}, "Chunk data does not match"
