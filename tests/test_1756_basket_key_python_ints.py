# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression test for issue #1756.

``TBranch.basket_key`` must request byte ranges with Python ints, since some
sources (e.g. XRootD through fsspec) reject NumPy integers. It is reached by
memory-based ``step_size`` through ``basket_uncompressed_bytes``.
"""

import skhep_testdata

import uproot


def test_iterate_memory_step_size_requests_python_ints(monkeypatch):
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as tree:
        source = tree.file.source
        original = source.chunk
        requested = []

        def spy(start, stop):
            requested.append((start, stop))
            return original(start, stop)

        monkeypatch.setattr(source, "chunk", spy)

        for _ in tree.iterate(["px1"], step_size="100 MB", library="np"):
            pass

        assert len(requested) > 0
        for start, stop in requested:
            assert type(start) is int
            assert type(stop) is int
