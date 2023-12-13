# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest
import uproot
import uproot.source.fsspec
import uproot.source.file
import uproot.source.xrootd
import uproot.source.s3

from typing import BinaryIO
import skhep_testdata
import queue
import fsspec
import requests
import os
import sys


@pytest.mark.parametrize(
    "urlpath, source_class",
    [
        ("file.root", uproot.source.fsspec.FSSpecSource),
        ("s3://path/file.root", uproot.source.fsspec.FSSpecSource),
        (r"C:\path\file.root", uproot.source.fsspec.FSSpecSource),
        (r"file://C:\path\file.root", uproot.source.fsspec.FSSpecSource),
        ("root://file.root", uproot.source.fsspec.FSSpecSource),
        (BinaryIO(), uproot.source.object.ObjectSource),
    ],
)
def test_default_source(urlpath, source_class):
    assert uproot._util.file_path_to_source_class(
        urlpath, options=uproot.reading.open.defaults
    ) == (source_class, urlpath)


@pytest.mark.parametrize(
    "to_open, handler",
    [
        ("file.root", "invalid_handler"),
        (BinaryIO(), uproot.source.fsspec.FSSpecSource),
    ],
)
def test_invalid_handler(to_open, handler):
    with pytest.raises(TypeError):
        uproot.open(to_open, handler=handler)


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
def test_open_fsspec_github():
    try:
        with uproot.open(
            "github://scikit-hep:scikit-hep-testdata@v0.4.33/src/skhep_testdata/data/uproot-issue121.root"
        ) as f:
            data = f["Events/MET_pt"].array(library="np")
            assert len(data) == 40
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            pytest.skip("GitHub API limit has been reached")
        else:
            raise e


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
        uproot.source.fsspec.FSSpecSource,
        uproot.source.s3.S3Source,
        None,
    ],
)
def test_open_fsspec_s3(handler):
    pytest.importorskip("s3fs")
    if sys.version_info < (3, 11):
        pytest.skip(
            "https://github.com/scikit-hep/uproot5/pull/1012",
        )

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
    pytest.importorskip("fsspec_xrootd")
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


def test_fsspec_memory():
    # read the file into memory
    with open(skhep_testdata.data_path("uproot-issue121.root"), "rb") as f:
        contents = f.read()

    # create a memory filesystem
    fs = fsspec.filesystem(protocol="memory")
    fs.store.clear()
    file_path = "skhep_testdata/uproot-issue121.root"
    fs.touch(file_path)
    # write contents into memory filesystem
    with fs.open(file_path, "wb") as f:
        f.write(contents)

    # read from memory filesystem
    with uproot.open(f"memory://{file_path}") as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


def test_fsspec_tar(tmp_path):
    import tarfile
    import io

    filename = "uproot-issue121.root"
    with open(skhep_testdata.data_path("uproot-issue121.root"), "rb") as f:
        contents = f.read()

    filename_tar = os.path.join(tmp_path, filename + ".tar")
    with tarfile.open(filename_tar, mode="w") as tar:
        file_info = tarfile.TarInfo(name=filename)
        file_info.size = len(contents)
        tar.addfile(file_info, fileobj=io.BytesIO(contents))

    # open with fsspec
    with uproot.open(f"tar://{filename}::file://{filename_tar}") as f:
        data = f["Events/MET_pt"].array(library="np")
        assert len(data) == 40


def test_fsspec_zip(tmp_path):
    import zipfile

    filename = "uproot-issue121.root"
    with open(skhep_testdata.data_path("uproot-issue121.root"), "rb") as f:
        contents = f.read()

    filename_zip = os.path.join(tmp_path, filename + ".zip")
    with zipfile.ZipFile(filename_zip, mode="w") as zip_file:
        zip_file.writestr(filename, data=contents)

    # open with fsspec
    with uproot.open(
        f"zip://{filename}:Events/MET_pt::file://{filename_zip}"
    ) as branch:
        data = branch.array(library="np")
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
def test_open_fsspec_xrootd_iterate_files(handler):
    pytest.importorskip("XRootD")

    iterator = uproot.iterate(
        files=[
            {
                "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root": "Events"
            },
            {
                "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root": "Events"
            },
        ],
        expressions=["run"],
        step_size=100,
        handler=handler,
    )

    for i, data in enumerate(iterator):
        if i >= 5:
            break
        assert len(data) == 100
        assert all(data["run"] == 194778)


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
def test_open_fsspec_xrootd_iterate_tree(handler):
    pytest.importorskip("XRootD")

    with uproot.open(
        {
            "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root": "Events"
        },
        handler=handler,
    ) as f:
        iterator = f.iterate(
            ["run"],
            step_size=100,
        )

        for i, data in enumerate(iterator):
            if i >= 5:
                break
            assert len(data) == 100
            assert all(data["run"] == 194778)


# https://github.com/scikit-hep/uproot5/issues/1035
@pytest.mark.parametrize(
    "handler",
    [
        uproot.source.file.MemmapSource,
        uproot.source.file.MultithreadedFileSource,
        uproot.source.fsspec.FSSpecSource,
        None,
    ],
)
def test_issue_1035(handler):
    with uproot.open(
        skhep_testdata.data_path("uproot-issue-798.root"),
        handler=handler,
        use_threads=True,
        num_workers=10,
    ) as f:
        for _ in range(25):  # intermittent failure
            tree = f["CollectionTree"]
            branch = tree["MuonSpectrometerTrackParticlesAuxDyn.truthParticleLink"]
            data = branch.array()
            assert len(data) == 40
