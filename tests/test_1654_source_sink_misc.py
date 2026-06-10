# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for PR #1654: source/sink error paths, leaks, and cleanups."""

from __future__ import annotations

import os
import time

import numpy
import pytest

import uproot
import uproot.sink.file
import uproot.source.chunk
import uproot.source.cursor
import uproot.source.futures
from uproot.source.coalesce import CoalesceConfig, RangeRequest, _coalesce


class _StubSource:
    file_path = "stub://file"


def _make_chunk(data: bytes) -> uproot.source.chunk.Chunk:
    raw = numpy.frombuffer(data, dtype=uproot.source.chunk.Chunk._dtype)
    future = uproot.source.futures.TrivialFuture(raw)
    return uproot.source.chunk.Chunk(_StubSource(), 0, len(data), future)


def test_classname_no_terminator_raises_oserror():
    # No NUL byte anywhere: must raise OSError (not IndexError/AttributeError)
    chunk = _make_chunk(b"abcdef")
    cursor = uproot.source.cursor.Cursor(0)
    with pytest.raises(OSError, match="no terminator"):
        cursor.classname(chunk, {})


def test_classname_terminator_at_last_byte():
    # Terminator at the final position previously triggered an IndexError
    chunk = _make_chunk(b"abc\x00")
    cursor = uproot.source.cursor.Cursor(0)
    assert cursor.classname(chunk, {}) == "abc"
    assert cursor.index == 4


def test_classname_reads_string_and_moves():
    chunk = _make_chunk(b"hello\x00world\x00")
    cursor = uproot.source.cursor.Cursor(0)
    assert cursor.classname(chunk, {}) == "hello"
    assert cursor.index == 6
    assert cursor.classname(chunk, {}) == "world"


def test_future_result_timeout_raises():
    # A task that never completes: result(timeout=...) must raise TimeoutError
    future = uproot.source.futures.Future(lambda: None, ())
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        future.result(timeout=0.05)
    assert time.monotonic() - start < 5.0


def test_future_result_reraises_exception():
    def boom():
        raise ValueError("kaboom")

    future = uproot.source.futures.Future(boom, ())
    future._run()
    with pytest.raises(ValueError, match="kaboom"):
        future.result()


def test_threadpool_shutdown_joins_workers():
    executor = uproot.source.futures.ThreadPoolExecutor(max_workers=3)
    workers = list(executor.workers)
    executor.shutdown()
    for worker in workers:
        assert not worker.is_alive()


def test_filesink_close_is_permanent(tmp_path):
    path = os.path.join(tmp_path, "sink.bin")
    sink = uproot.sink.file.FileSink(path)
    assert sink.closed is False  # not opened yet, but not closed either
    sink.write(0, b"hello")
    sink.close()
    assert sink.closed is True
    # After close, operations must raise instead of silently reopening
    with pytest.raises(OSError, match="closed"):
        sink.write(0, b"world")
    with pytest.raises(OSError, match="closed"):
        sink.tell()
    with pytest.raises(OSError, match="closed"):
        sink.read(0, 1)


def test_coalesce_batches_at_max_request_ranges():
    # Distinct, well-separated ranges so each is its own cluster.
    config = CoalesceConfig(
        max_range_gap=0,
        max_request_ranges=4,
        max_request_bytes=10 * 1024 * 1024,
        min_first_request_bytes=10 * 1024 * 1024,
    )
    ranges = [(10 * i, 10 * i + 1) for i in range(8)]
    all_requests = [RangeRequest(start, stop, None) for start, stop in ranges]
    batches = list(_coalesce(all_requests, config))
    # 8 clusters with max 4 per request -> exactly 2 full batches
    assert [len(batch.clusters) for batch in batches] == [4, 4]
