# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

import pytest
import threading
import contextlib
import skhep_testdata
from functools import partial

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
    # serve files from the skhep_testdata data directory
    handler = partial(
        RangeRequestHandler, directory=skhep_testdata.local_files._cache_path()
    )
    server = HTTPServer(server_address=("localhost", 0), RequestHandlerClass=handler)
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
def server():
    with serve() as server_url:
        yield server_url
