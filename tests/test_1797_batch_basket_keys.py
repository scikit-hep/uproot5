# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
A memory-based ``step_size`` must read all ``TBasket`` ``TKeys`` in one
request, not one round trip per basket.
"""

import skhep_testdata

import uproot


def test_num_entries_for_reads_keys_in_one_request(monkeypatch):
    with uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"] as tree:
        expected_bytes = sum(
            branch.basket_uncompressed_bytes(i)
            for branch in tree.itervalues(recursive=True)
            for i in range(branch.num_baskets)
        )
        num_baskets = sum(
            branch.num_baskets for branch in tree.itervalues(recursive=True)
        )
        assert num_baskets > 1

        source = tree.file.source
        calls = {"chunk": 0, "chunks": 0}
        original_chunk = source.chunk
        original_chunks = source.chunks

        def chunk(start, stop):
            calls["chunk"] += 1
            return original_chunk(start, stop)

        def chunks(ranges, notifications):
            calls["chunks"] += 1
            assert all(type(x) is int for r in ranges for x in r)
            return original_chunks(ranges, notifications)

        monkeypatch.setattr(source, "chunk", chunk)
        monkeypatch.setattr(source, "chunks", chunks)

        result = tree.num_entries_for(expected_bytes)

        assert calls == {"chunk": 0, "chunks": 1}
        assert result == tree.num_entries
