# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""Regression tests for issue #1688: executor, chunk, and unknown-class fixes.

* a submit racing with shutdown was accepted and then orphaned behind a sentinel
* ``ThreadPoolExecutor.submit`` accepted keyword arguments and discarded them
* a Chunk that failed its length check returned the short buffer on re-access
* unknown model classes were cached by classname alone, ignoring the version
"""

from __future__ import annotations

import threading
import time

import pytest

import uproot
import uproot._util
import uproot.model
import uproot.source.chunk
import uproot.source.futures


class DummyResource:
    file_path = "dummy"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_submit_during_shutdown_is_rejected_not_orphaned():
    executor = uproot.source.futures.ResourceThreadPoolExecutor([DummyResource()])
    executor.__enter__()

    # occupy the only worker so that shutdown's join() blocks for a while
    executor.submit(
        uproot.source.futures.ResourceFuture(lambda resource: time.sleep(1))
    )
    time.sleep(0.1)

    closer = threading.Thread(target=lambda: executor.__exit__(None, None, None))
    closer.start()
    try:
        time.sleep(0.1)  # the sentinel is queued and join() is blocking
        assert executor.closed
        late = uproot.source.futures.ResourceFuture(lambda resource: 42)
        with pytest.raises(OSError):
            executor.submit(late)
    finally:
        closer.join()


def test_thread_pool_executor_rejects_submit_after_shutdown():
    executor = uproot.source.futures.ThreadPoolExecutor(1)
    assert not executor.closed
    executor.shutdown()
    assert executor.closed
    with pytest.raises(OSError):
        executor.submit(lambda: 1)


def test_thread_pool_executor_forwards_kwargs():
    executor = uproot.source.futures.ThreadPoolExecutor(1)
    try:
        future = executor.submit(lambda a, x=None: (a, x), 1, x=3)
        assert future.result(timeout=10) == (1, 3)

        future = executor.submit(lambda *, only_kw: only_kw, only_kw="value")
        assert future.result(timeout=10) == "value"
    finally:
        executor.shutdown()


def test_trivial_executor_forwards_kwargs():
    executor = uproot.source.futures.TrivialExecutor()
    future = executor.submit(lambda a, x=None: (a, x), 1, x=3)
    assert future.result() == (1, 3)


class _FakeSource:
    file_path = "fake"


def test_short_chunk_raises_every_time():
    chunk = uproot.source.chunk.Chunk(
        _FakeSource(), 0, 5, uproot.source.futures.TrivialFuture(b"abc")
    )
    for _ in range(3):
        with pytest.raises(OSError, match="expected Chunk of length 5"):
            chunk.raw_data


def test_chunk_of_expected_length_still_works():
    chunk = uproot.source.chunk.Chunk(
        _FakeSource(), 0, 5, uproot.source.futures.TrivialFuture(b"abcde")
    )
    assert chunk.raw_data.tobytes() == b"abcde"
    assert chunk.raw_data.tobytes() == b"abcde"


def test_chunk_insist_false_does_not_raise():
    chunk = uproot.source.chunk.Chunk(
        _FakeSource(), 0, 5, uproot.source.futures.TrivialFuture(b"abc")
    )
    chunk.wait(insist=False)
    assert chunk.raw_data.tobytes() == b"abc"


class _StreamerlessFile:
    custom_classes = None
    file_path = "fake"
    streamers = {}

    def streamer_named(self, classname, version):
        return None


def _make_dispatch():
    return uproot._util.new_class(
        uproot.model.classname_encode("MyClass"),
        (uproot.model.DispatchByVersion,),
        {"known_versions": {}},
    )


def test_unknown_class_versions_are_distinct():
    uproot.model.reset_classes()
    try:
        dispatch = _make_dispatch()

        v1 = dispatch.new_class(_StreamerlessFile(), 1)
        v2 = dispatch.new_class(_StreamerlessFile(), 2)

        assert v1.__name__ == "Unknown_MyClass_v1"
        assert v2.__name__ == "Unknown_MyClass_v2"
        assert v1 is not v2
        # asking again returns the cached class rather than building a new one
        assert dispatch.new_class(_StreamerlessFile(), 1) is v1
        assert set(uproot.unknown_classes) == {
            "Unknown_MyClass_v1",
            "Unknown_MyClass_v2",
        }
    finally:
        uproot.model.reset_classes()


class _NoStreamerFile:
    """Enough of a ReadOnlyFile for class_named to reach the unknown-class paths."""

    def __init__(self, custom_classes=None):
        self._custom_classes = custom_classes

    def streamers_named(self, classname):
        return []

    def streamer_named(self, classname, version):
        return None


def test_versioned_and_versionless_unknown_classes_coexist():
    uproot.model.reset_classes()
    try:
        dispatch = _make_dispatch()
        # a file that knows the class but not the requested version
        versioned = uproot.reading.ReadOnlyFile.class_named(
            _NoStreamerFile({"MyClass": dispatch}), "MyClass", "max"
        )
        # a file with no streamers at all for the same class
        versionless = uproot.reading.ReadOnlyFile.class_named(
            _NoStreamerFile(), "MyClass"
        )

        assert issubclass(versioned, uproot.model.UnknownClassVersion)
        assert issubclass(versionless, uproot.model.UnknownClass)
        assert versioned is not versionless
        assert versionless.__name__ == "Unknown_MyClass"
    finally:
        uproot.model.reset_classes()
