# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
fsspec's cat_ranges returns exceptions in its result list by default. A read
must raise that exception, not a TypeError from slicing it.
"""

import pytest

from uproot.source.coalesce import CoalesceConfig, Future, RangeRequest, _coalesce


def test_coalesce_raises_returned_exception():
    data = b"abcdefghijklmnopqurstuvwxyz"
    error = ConnectionResetError("Connection reset by peer")

    ranges = [(1, 3), (4, 6), (20, 25)]
    all_requests = [RangeRequest(start, stop, None) for start, stop in ranges]
    for merged_request in _coalesce(all_requests, CoalesceConfig(max_range_gap=2)):
        future = Future()
        future.set_result(
            [
                error if start >= 20 else data[start:stop]
                for start, stop in merged_request.ranges()
            ]
        )
        merged_request.set_future(future)

    good1, good2, bad = all_requests
    assert good1.future.result() == data[1:3]
    assert good2.future.result() == data[4:6]
    with pytest.raises(ConnectionResetError, match="reset by peer"):
        bad.future.result()
