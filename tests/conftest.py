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

    server = HTTPServer(
        server_address=("localhost", 0),
        RequestHandlerClass=partial(
            Handler, directory=skhep_testdata.local_files._cache_path()
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


@pytest.fixture(scope="module")
def xrootd_server(tmpdir_factory):
    pytest.importorskip("XRootD")
    pytest.importorskip("fsspec_xrootd")

    server_dir = tmpdir_factory.mktemp("server")
    temp_path = os.path.join(server_dir, "Folder")
    os.mkdir(temp_path)
    xrootd = shutil.which("xrootd")
    proc = subprocess.Popen([xrootd, server_dir])
    time.sleep(2)  # give it some startup
    yield "root://localhost/" + str(temp_path), temp_path
    proc.terminate()
    proc.wait(timeout=10)
    shutil.rmtree(server_dir)
