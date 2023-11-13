# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import pytest
import uproot
import uproot.source.fsspec

import os
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


@pytest.mark.parametrize("scheme", ["", "file://"])
def test_fsspec_writing_local(tmp_path, scheme):
    uri = scheme + os.path.join(tmp_path, "some", "path", "file.root")
    with uproot.recreate(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]


def test_fsspec_writing_http(server):
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
        # "file://",  # This fails because of the fsspec file-like object cannot be used for reading and writing at the same time
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


@pytest.mark.skip(reason="not working yet")
def test_fsspec_writing_memory(tmp_path):
    uri = f"memory://{tmp_path}/file.root"

    with uproot.recreate(uri) as f:
        f["tree"] = {"x": np.array([1, 2, 3])}

    with uproot.open(uri) as f:
        assert f["tree"]["x"].array().tolist() == [1, 2, 3]
