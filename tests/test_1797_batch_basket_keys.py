# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
A memory-based ``step_size`` must read all ``TBasket`` ``TKeys`` in one
request, not one round trip per basket.
"""

from __future__ import annotations

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

        def chunks(ranges, notifications, **kwargs):
            calls["chunks"] += 1
            assert all(type(x) is int for r in ranges for x in r)
            return original_chunks(ranges, notifications, **kwargs)

        monkeypatch.setattr(source, "chunk", chunk)
        monkeypatch.setattr(source, "chunks", chunks)

        result = tree.num_entries_for(expected_bytes)

        assert calls == {"chunk": 0, "chunks": 1}
        assert result == tree.num_entries


def test_num_entries_for_fsspec_does_not_coalesce_keys():
    path = skhep_testdata.data_path("uproot-HZZ.root")
    with uproot.open(f"file://{path}", handler=uproot.source.fsspec.FSSpecSource)[
        "events"
    ] as tree:
        branches = list(tree.itervalues(recursive=True))
        num_baskets = sum(branch.num_baskets for branch in branches)
        assert num_baskets > 1
        expected_bytes = sum(
            branch.basket_uncompressed_bytes(i)
            for branch in branches
            for i in range(branch.num_baskets)
        )

        source = tree.file.source
        fetched = []
        original_cat_ranges = source._fs.cat_ranges

        def cat_ranges(paths, starts, ends, **kwargs):
            fetched.extend(stop - start for start, stop in zip(starts, ends))
            return original_cat_ranges(paths, starts, ends, **kwargs)

        source._fs.cat_ranges = cat_ranges
        source._async_impl = False

        assert tree.num_entries_for(expected_bytes) == tree.num_entries
        assert sum(fetched) == num_baskets * uproot.reading._key_format_big.size


def test_num_entries_for_entry_range():
    path = skhep_testdata.data_path("uproot-sample-6.20.04-uncompressed.root")
    with uproot.open(path)["sample"] as tree:
        branch = tree["n"]
        assert branch.num_baskets > 2
        start, stop = branch.entry_offsets[1], branch.entry_offsets[2]
        expected_bytes = sum(
            b.basket_uncompressed_bytes(i)
            for b in tree.itervalues(recursive=True)
            for i in range(b.num_baskets)
            if b.entry_offsets[i] < stop and start < b.entry_offsets[i + 1]
        )
        assert (
            tree.num_entries_for(expected_bytes, entry_start=start, entry_stop=stop)
            == stop - start
        )
