# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import asyncio
import concurrent.futures
import queue

import fsspec
import fsspec.asyn

import uproot
import uproot.source.chunk
import uproot.source.futures


class FSSpecSource(uproot.source.chunk.Source):
    """
    Args:
        file_path (str): A URL for the file to open.
        **kwargs (dict): any extra arguments to be forwarded to the particular
            FileSystem instance constructor. This might include S3 access keys,
            or HTTP headers, etc.

    A :doc:`uproot.source.chunk.Source` that uses FSSpec's cat_ranges feature
    to get many chunks in one request.
    """

    def __init__(self, file_path: str, **options):
        options = dict(uproot.reading.open.defaults, **options)
        storage_options = {
            k: v
            for k, v in options.items()
            if k not in uproot.reading.open.defaults.keys()
        }

        self._fs, self._file_path = fsspec.core.url_to_fs(file_path, **storage_options)

        # What should we do when there is a chain of filesystems?
        self._async_impl = self._fs.async_impl

        self._executor = None
        self._file = None
        self._fh = None

        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._open()

        self.__enter__()

    def _open(self):
        self._executor = FSSpecLoopExecutor()
        self._file = self._fs.open(self._file_path)

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        return f"<{type(self).__name__} {path} at 0x{id(self):012x}>"

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_executor")
        state.pop("_file")
        state.pop("_fh")
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open()

    def __enter__(self):
        self._fh = self._file.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._fh = None
        self._file.__exit__(exception_type, exception_value, traceback)
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
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start
        if self._fh:
            self._fh.seek(start)
            data = self._fh.read(stop - start)
        else:
            data = self._fs.cat_file(self._file_path, start, stop)
        future = uproot.source.futures.TrivialFuture(data)
        return uproot.source.chunk.Chunk(self, start, stop, future)

    def chunks(
        self, ranges: list[(int, int)], notifications: queue.Queue
    ) -> list[uproot.source.chunk.Chunk]:
        """
        Args:
            ranges (list of (int, int) 2-tuples): Intervals to fetch
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
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        try:
            # not available in python 3.8
            to_thread = asyncio.to_thread
        except AttributeError:
            import contextvars
            import functools

            async def to_thread(func, /, *args, **kwargs):
                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                func_call = functools.partial(ctx.run, func, *args, **kwargs)
                return await loop.run_in_executor(None, func_call)

        async def async_wrapper_thread(blocking_func, *args, **kwargs):
            if not callable(blocking_func):
                raise TypeError("blocking_func must be callable")
            # TODO: when python 3.8 is dropped, use `asyncio.to_thread` instead (also remove the try/except block above)
            return await to_thread(blocking_func, *args, **kwargs)

        chunks = []
        for start, stop in ranges:
            # _cat_file is async while cat_file is not.
            coroutine = (
                self._fs._cat_file(self._file_path, start=start, end=stop)
                if self._async_impl
                else async_wrapper_thread(
                    self._fs.cat_file, self._file_path, start=start, end=stop
                )
            )

            future = self._executor.submit(coroutine)

            chunk = uproot.source.chunk.Chunk(self, start, stop, future)
            future.add_done_callback(uproot.source.chunk.notifier(chunk, notifications))
            chunks.append(chunk)
        return chunks

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
        return False


class FSSpecLoopExecutor(uproot.source.futures.Executor):
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return fsspec.asyn.get_loop()

    def submit(self, coroutine) -> concurrent.futures.Future:
        if not asyncio.iscoroutine(coroutine):
            raise TypeError("loop executor can only submit coroutines")
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)
