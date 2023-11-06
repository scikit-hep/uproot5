# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest
import uproot
import uproot.source.fsspec

import skhep_testdata
import queue
import os
import fsspec
import numpy as np


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


@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_open_fsspec_ssh(handler):
    pytest.importorskip("paramiko")
    import paramiko
    import getpass

    user = getpass.getuser()
    host = "localhost"
    port = 22

    # only test this if we can connect to the host (this will work in GitHub Actions)
    try:
        with paramiko.SSHClient() as client:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, port=port, username=user)
    except (
        paramiko.ssh_exception.SSHException,
        paramiko.ssh_exception.NoValidConnectionsError,
    ) as e:
        pytest.skip(f"ssh connection to host failed: {e}")

    # cache the file
    local_path = skhep_testdata.data_path("uproot-issue121.root")

    uri = f"ssh://{user}@{host}:{port}{local_path}"
    with uproot.open(uri, handler=handler) as f:
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


def test_fsspec_writing_no_integration(tmp_path):
    uri = f"file://{tmp_path}/file.root"
    with fsspec.open(uri, mode="wb") as file_obj:
        # write a simple root file
        with uproot.recreate(file_obj) as f:
            f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


def test_fsspec_writing_local(tmp_path):
    uri = f"file://{tmp_path}/file.root"
    with uproot.recreate(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


def test_fsspec_writing_ssh(tmp_path):
    pytest.importorskip("paramiko")
    import paramiko
    import getpass

    user = getpass.getuser()
    host = "localhost"
    port = 22

    # only test this if we can connect to the host (this will work in GitHub Actions)
    try:
        with paramiko.SSHClient() as client:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, port=port, username=user)
    except (
        paramiko.ssh_exception.SSHException,
        paramiko.ssh_exception.NoValidConnectionsError,
    ) as e:
        pytest.skip(f"ssh connection to host failed: {e}")

    local_path = os.path.join(tmp_path, "file.root")
    uri = f"ssh://{user}@{host}:{port}{local_path}"

    with uproot.recreate(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


def test_fsspec_writing_memory(tmp_path):
    uri = f"memory://{tmp_path}/file.root"

    with uproot.recreate(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]
