# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest
import uproot
import uproot.source.fsspec

import os
import pathlib
import fsspec
import numpy as np


def test_fsspec_writing_no_integration(tmp_path):
    uri = os.path.join(tmp_path, "some", "path", "file.root")
    with fsspec.open(uri, mode="wb") as file_obj:
        # write a simple root file
        with uproot.recreate(file_obj) as f:
            f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


@pytest.mark.parametrize("scheme", ["", "file://", "simplecache::file://"])
def test_fsspec_writing_local(tmp_path, scheme):
    uri = scheme + os.path.join(tmp_path, "some", "path", "file.root")
    with uproot.recreate(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


# https://github.com/scikit-hep/uproot5/issues/325
@pytest.mark.parametrize(
    "scheme",
    [
        "",
        "file:",  # this is not a valid schema, but we should support it
        "file://",
        "simplecache::file://",
    ],
)
@pytest.mark.parametrize(
    "filename", ["file.root", "file%2Eroot", "my%E2%80%92file.root", "my%20file.root"]
)
@pytest.mark.parametrize(
    "slash_prefix",
    ["", "/", "\\"],
)
def test_fsspec_writing_local_uri(tmp_path, scheme, slash_prefix, filename):
    uri = scheme + slash_prefix + os.path.join(tmp_path, "some", "path", filename)
    print(uri)
    with uproot.create(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}
    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "scheme",
    [
        "",
        "file://",
        "simplecache::file://",
        # "memory://", # uncomment when https://github.com/fsspec/filesystem_spec/pull/1426 is available in pypi
        "simplecache::memory://",
    ],
)
def test_fsspec_writing_create(tmp_path, scheme):
    uri = scheme + os.path.join(tmp_path, "some", "path", "file.root")
    with uproot.create(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}

    with pytest.raises(FileExistsError):
        with uproot.create(uri):
            pass


def test_issue_1029(tmp_path):
    # https://github.com/scikit-hep/uproot5/issues/1029
    urlpath = os.path.join(tmp_path, "some", "path", "file.root")
    urlpath = pathlib.Path(urlpath)

    with uproot.recreate(urlpath) as f:
        f["tree_1"] = {"x": np.array([1, 2, 3])}

    with uproot.update(urlpath) as f:
        f["tree_2"] = {"y": np.array([4, 5, 6])}

    with uproot.open(urlpath) as f:
        assert f["tree_1"]["x"].array().tolist() == [1, 2, 3]
        assert f["tree_2"]["y"].array().tolist() == [4, 5, 6]


def test_fsspec_writing_http(server):
    pytest.importorskip("aiohttp")

    uri = f"{server}/file.root"

    with pytest.raises(NotImplementedError):
        # TODO: review this when fsspec supports writing to http
        with uproot.recreate(uri) as f:
            f["tree"] = {"x": np.array([1, 2, 3])}

        with uproot.open(uri) as f:
            assert f["tree"]["x"].array().tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "scheme",
    [
        "",
        "file://",
        "simplecache::file://",
    ],
)
def test_fsspec_writing_local_update(tmp_path, scheme):
    uri = scheme + os.path.join(tmp_path, "some", "path", "file.root")
    with uproot.recreate(uri) as f:
        f["tree1"] = {"x": np.array([1, 2, 3])}

    with uproot.update(uri) as f:
        f["tree2"] = {"y": np.array([4, 5, 6])}

    # read data and compare
    with uproot.open(uri) as f:
        assert f["tree1"]["x"].array().tolist() == [1, 2, 3]
        assert f["tree2"]["y"].array().tolist() == [4, 5, 6]


@pytest.mark.parametrize(
    "scheme",
    [
        "ssh://",
        "simplecache::ssh://",
    ],
)
def test_fsspec_writing_ssh(tmp_path, scheme):
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
    uri = f"{scheme}{user}@{host}:{port}{local_path}"

    with uproot.recreate(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "scheme",
    [
        "memory://",
        "simplecache::memory://",
    ],
)
def test_fsspec_writing_memory(tmp_path, scheme):
    uri = f"{scheme}{tmp_path}/file.root"

    # when https://github.com/fsspec/filesystem_spec/pull/1426 is merged this will work, we must remove the catch
    with pytest.raises(ValueError):
        with uproot.recreate(uri) as f:
            f["tree"] = {"x": np.array([1, 2, 3])}

        with uproot.open(uri) as f:
            assert f["tree"]["x"].array().tolist() == [1, 2, 3]

        if scheme == "simplecache::memory://":
            raise ValueError("wait for next fsspec release and remove this")
