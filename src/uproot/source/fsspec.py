# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

from __future__ import annotations

import asyncio
import concurrent.futures
import queue

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
        import fsspec.core

        default_options = uproot.reading.open.defaults
        self._use_threads = options.get("use_threads", default_options["use_threads"])
        self._num_workers = options.get("num_workers", default_options["num_workers"])

        # TODO: is timeout always valid?

        # Remove uproot-specific options (should be done earlier)
        exclude_keys = set(default_options.keys())
        storage_options = {k: v for k, v in options.items() if k not in exclude_keys}

        protocol = fsspec.core.split_protocol(file_path)[0]
        fs_has_async_impl = fsspec.get_filesystem_class(protocol=protocol).async_impl

        if self._use_threads:
            if fs_has_async_impl:
                self._executor = FSSpecLoopExecutor(fsspec.asyn.get_loop())
            else:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._num_workers
                )
        else:
            self._executor = uproot.source.futures.TrivialExecutor()

        self._fs, self._file_path = fsspec.core.url_to_fs(file_path, **storage_options)

        # TODO: set mode to "read-only" in a way that works for all filesystems
        self._file = self._fs.open(self._file_path)
        self._fh = None
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0
        self.__enter__()

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

        chunks = []
        # _cat_file is async while cat_file is not
        use_async = isinstance(self._executor, FSSpecLoopExecutor)
        cat_file = self._fs._cat_file if use_async else self._fs.cat_file
        for start, stop in ranges:
            future = self._executor.submit(cat_file, self._file_path, start, stop)
            chunk = uproot.source.chunk.Chunk(self, start, stop, future)
            future.add_done_callback(uproot.source.chunk.notifier(chunk, notifications))
            chunks.append(chunk)
        return chunks

    @property
    def num_bytes(self) -> int:
        """
        The number of bytes in the file.
        """
        return self._fs.size(self._file_path)

    @property
    def closed(self) -> bool:
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return False


class FSSpecLoopExecutor:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def submit(self, coroutine, *args) -> asyncio.Future:
        if not asyncio.iscoroutinefunction(coroutine):
            raise TypeError("loop executor can only submit coroutines")
        if not self.loop.is_running():
            raise RuntimeError("cannot submit coroutine while loop is not running")
        coroutine_object = coroutine(*args)
        return asyncio.run_coroutine_threadsafe(coroutine_object, self.loop)

    def shutdown(self, wait=True):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
