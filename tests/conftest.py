# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import threading
import contextlib
import os
import skhep_testdata

# The base http server does not support range requests. Watch https://github.com/python/cpython/issues/86809 for updates
from http.server import HTTPServer
from RangeHTTPServer import RangeRequestHandler

import uproot


@pytest.fixture(scope="function", autouse=False)
def reset_classes():
    uproot.model.reset_classes()
    return


@contextlib.contextmanager
def serve():
    port = 7879  # any available port
    server_address = ("", port)
    httpd = HTTPServer(server_address, RangeRequestHandler)
    # serve files from the skhep_testdata data directory
    files_directory = skhep_testdata.local_files._cache_path()
    os.chdir(files_directory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        try:
            httpd.socket.close()
            httpd.shutdown()
            thread.join()
        except Exception as e:
            print("Exception while shutting down server", e)


@pytest.fixture(scope="module")
def server():
    with serve() as server_url:
        yield server_url
