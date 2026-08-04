# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
import shutil
import subprocess
import pytest
import threading
import contextlib
import skhep_testdata
from functools import partial
import os
import time

# The base http server does not support range requests. Watch https://github.com/python/cpython/issues/86809 for updates
from http.server import HTTPServer
from RangeHTTPServer import RangeRequestHandler

import uproot


@pytest.fixture(scope="function", autouse=False)
def reset_classes():
    uproot.model.reset_classes()
    return


@contextlib.contextmanager
def serve_http():
    # serve files from the skhep_testdata cache directory.
    # This directory is initially empty and files are downloaded on demand
    class Handler(RangeRequestHandler):
        def _cache_file(self, path: str):
            path = path.lstrip("/")
            if path in skhep_testdata.known_files:
                return skhep_testdata.data_path(path)
            else:
                raise FileNotFoundError(
                    f"File '{path}' not available in skhep_testdata"
                )

        def do_HEAD(self):
            self._cache_file(self.path)
            return super().do_HEAD()

        def do_GET(self):
            self._cache_file(self.path)
            return super().do_GET()

    try:
        # Older skhep_testdata (in Python 3.9 environments)
        cache_path = skhep_testdata.local_files._cache_path()
    except AttributeError:
        # Newer skhep_testdata
        cache_path = skhep_testdata.data.cache_path()

    server = HTTPServer(
        server_address=("localhost", 0),
        RequestHandlerClass=partial(
            Handler,
            directory=cache_path,
        ),
    )
    server.server_activate()

    def serve_forever(httpd=server):
        with httpd:
            httpd.serve_forever()

    thread = threading.Thread(target=serve_forever, daemon=True)

    try:
        thread.start()
        address, port = server.server_address
        yield f"http://{address}:{port}"
    finally:
        # stop the server
        server.shutdown()
        thread.join()


@pytest.fixture(scope="module")
def http_server():
    with serve_http() as server_url:
        yield server_url


@pytest.fixture(scope="module")
def tests_directory() -> str:
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def s3_server():
    """
    A local (in-process) S3 server, so that the S3 tests don't depend on the
    network or on somebody else paying for the data transfer.

    Yields ``(bucket_url, storage_options)``, where the bucket already contains
    two copies of ``uproot-HZZ.root``, named ``uproot-HZZ-1.root`` and
    ``uproot-HZZ-2.root``.
    """
    pytest.importorskip("s3fs")
    moto_server = pytest.importorskip("moto.server")
    import s3fs

    if not hasattr(moto_server.ThreadedMotoServer, "get_host_and_port"):
        # On free-threaded Windows, moto[server] -> docker -> pywin32 has no
        # wheels, so the resolver falls back to a years-old moto
        pytest.skip("moto is too old to report which port it is listening on")

    bucket = "uproot-test"
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()

    try:
        _, port = server.get_host_and_port()
        storage_options = {
            # moto does not check these, but botocore insists on having them
            "key": "testing",
            "secret": "testing",
            "client_kwargs": {"endpoint_url": f"http://127.0.0.1:{port}"},
        }

        with pytest.MonkeyPatch.context() as monkeypatch:
            # botocore raises NoRegionError if it can't find a region anywhere
            monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

            # don't reuse (or poison) filesystem instances from other tests
            s3fs.S3FileSystem.clear_instance_cache()
            fs = s3fs.S3FileSystem(**storage_options)
            fs.mkdir(bucket)
            local_path = skhep_testdata.data_path("uproot-HZZ.root")
            for name in ("uproot-HZZ-1.root", "uproot-HZZ-2.root"):
                fs.put(local_path, f"{bucket}/{name}")

            yield f"s3://{bucket}", storage_options

            s3fs.S3FileSystem.clear_instance_cache()
    finally:
        server.stop()


@pytest.fixture(scope="module")
def xrootd_server(tmpdir_factory):
    pytest.importorskip("XRootD")
    pytest.importorskip("fsspec_xrootd")

    server_dir = tmpdir_factory.mktemp("server")
    temp_path = os.path.join(server_dir, "Folder")
    os.mkdir(temp_path)
    xrootd = shutil.which("xrootd")
    if xrootd is None:
        pytest.skip("xrootd server executable is not available on PATH")
    proc = subprocess.Popen([xrootd, server_dir])
    time.sleep(2)  # give it some startup
    yield "root://localhost/" + str(temp_path), temp_path
    proc.terminate()
    proc.wait(timeout=10)
    shutil.rmtree(server_dir)
