# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import re

import fsspec
import fsspec.asyn

import uproot
import uproot.source.chunk
import uproot.source.futures
from uproot.source.coalesce import CoalesceConfig, coalesce_requests

# Patterns for known problematic servers
_PATTERN_WEBDAV = re.compile(r"https?://.+/remote\.php/dav/public-files/.+")


def _maybe_wrap_remote_url(url: str) -> str:
    """
    Wrap remote URLs with simplecache:: if they match the patterns
    """
    # Skip local files and known good protocols
    if url.startswith(("/", "root://", "s3://", "gs://")):
        return url

    # First, check known problematic servers by regex
    if _PATTERN_WEBDAV.match(url):
        return f"simplecache::{url}"

    return url


class FSSpecSource(uproot.source.chunk.Source):
    """
    Args:
        file_path (str): A URL for the file to open.
        coalesce_config (struct, optional): Configuration options for read coalescing
        **kwargs (dict): any extra arguments to be forwarded to the particular
            FileSystem instance constructor. This might include S3 access keys,
            or HTTP headers, etc.

    A :doc:`uproot.source.chunk.Source` that uses FSSpec's cat_ranges feature
    to get many chunks in one request.
    """

    def __init__(
        self, file_path: str, coalesce_config: CoalesceConfig | None = None, **options
    ):
        super().__init__()
        self._coalesce_config = coalesce_config

        file_path = _maybe_wrap_remote_url(file_path)

        fsspec_options = {
            k: v for k, v in options.items() if k not in uproot.reading.open.defaults
        }
        self._fs, self._file_path = fsspec.core.url_to_fs(file_path, **fsspec_options)

        # What should we do when there is a chain of filesystems?
        self._async_impl = self._fs.async_impl

        self._open()

        self.__enter__()

    def _open(self):
        self._executor = FSSpecLoopExecutor()

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        return f"<{type(self).__name__} {path} at 0x{id(self):012x}>"

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_executor")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._executor.shutdown()

    def chunk(self, start: int, stop: int) -> uproot.source.chunk.Chunk:
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Request a byte range of data from the file as a
        :doc:`uproot.source.chunk.Chunk`.
        """
        if self.closed:
            raise OSError(f"file {self._file_path!r} is closed")

        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start
        data = self._fs.cat_file(self._file_path, start=start, end=stop)
        future = uproot.source.futures.TrivialFuture(data)
        return uproot.source.chunk.Chunk(self, start, stop, future)

    def chunks(
        self, ranges: list[tuple[int, int]], notifications: queue.Queue
    ) -> list[uproot.source.chunk.Chunk]:
        """
        Args:
            ranges (list of tuple[int, int] 2-tuples): Intervals to fetch
                as (start, stop) pairs in a single request, if possible.
            notifications (``queue.Queue``): Indicator of completed
                chunks. After each gets filled, it is ``put`` on the
                queue; a listener should ``get`` from this queue
                ``len(ranges)`` times.

        Request a set of byte ranges from the file.

        This method has two outputs:

        * The method returns a list of unfilled
          :doc:`uproot.source.chunk.Chunk` objects, which get filled
          in a background thread. If you try to read data from an
          unfilled chunk, it will wait until it is filled.
        * The method also puts the same :doc:`uproot.source.chunk.Chunk`
          objects onto the ``notifications`` queue as soon as they are
          filled.

        Reading data from chunks on the queue can be more efficient than
        reading them from the returned list. The total reading time is the
        same, but work on the filled chunks can be better parallelized if
        it is triggered by already-filled chunks, rather than waiting for
        chunks to be filled.
        """
        if self.closed:
            raise OSError(f"file {self._file_path!r} is closed")

        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        async def async_wrapper_thread(blocking_func, *args, **kwargs):
            if not callable(blocking_func):
                raise TypeError("blocking_func must be callable")
            return await asyncio.to_thread(blocking_func, *args, **kwargs)

        def submit(request_ranges: list[tuple[int, int]]):
            paths = [self._file_path] * len(request_ranges)
            starts = [start for start, _ in request_ranges]
            ends = [stop for _, stop in request_ranges]
            # _cat_ranges is async while cat_ranges is not.
            coroutine = (
                self._fs._cat_ranges(paths=paths, starts=starts, ends=ends)
                if self._async_impl and not self._fs._cached
                else async_wrapper_thread(
                    self._fs.cat_ranges, paths=paths, starts=starts, ends=ends
                )
            )
            if uproot._util.wasm:
                # Threads can't be spawned in pyodide yet, so we run the function directly
                # and return a future that is already resolved.
                return uproot.source.futures.TrivialFuture(
                    self._fs.cat_ranges(paths=paths, starts=starts, ends=ends)
                )
            return self._executor.submit(coroutine)

        return coalesce_requests(
            ranges, submit, self, notifications, config=self._coalesce_config
        )

    @property
    def async_impl(self) -> bool:
        """
        True if using an async loop executor; False otherwise.
        """
        return self._async_impl

    @property
    def num_bytes(self) -> int:
        """
        The number of bytes in the file.
        """
        return self._fs.size(self._file_path)

    # We need this because user may use other executors not defined in `uproot.source` (such as `concurrent.futures`)
    # that do not have this interface. If not defined it defaults to calling this property on the source's executor
    @property
    def closed(self) -> bool:
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        # FSSpecSource uses entirely stateless interfaces
        return False


class FSSpecLoopExecutor(uproot.source.futures.Executor):
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return fsspec.asyn.get_loop()

    def submit(self, coroutine) -> concurrent.futures.Future:
        if not asyncio.iscoroutine(coroutine):
            raise TypeError("loop executor can only submit coroutines")
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)
