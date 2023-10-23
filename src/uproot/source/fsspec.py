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
        # Add the possibility to set use_async directly as a hidden option.
        # It is not encouraged to do so but may be useful for testing purposes.
        self._use_async = options.get("use_async", None) if self._use_threads else False

        # TODO: is timeout always valid?

        # Remove uproot-specific options (should be done earlier)
        exclude_keys = set(default_options.keys())
        storage_options = {k: v for k, v in options.items() if k not in exclude_keys}

        protocol = fsspec.core.split_protocol(file_path)[0]
        fs_has_async_impl = fsspec.get_filesystem_class(protocol=protocol).async_impl
        # If not explicitly set (default), use async if possible
        self._use_async = (
            fs_has_async_impl if self._use_async is None else self._use_async
        )
        if self._use_async and not fs_has_async_impl:
            # This should never be triggered unless the user explicitly set the `use_async` flag for a non-async backend
            raise ValueError(f"Filesystem {protocol} does not support async")

        self._fs, self._file_path = fsspec.core.url_to_fs(file_path, **storage_options)

        if not self._use_threads:
            self._executor = uproot.source.futures.TrivialExecutor()
        elif self._use_async:
            self._executor = FSSpecLoopExecutor()
            try:
                import s3fs.core

                if isinstance(self._fs, s3fs.core.S3FileSystem):
                    self._session = asyncio.run_coroutine_threadsafe(
                        self._fs.set_session(), self._executor.loop
                    ).result()
            except ImportError:
                ...
        else:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_workers
            )

        if not self._use_threads:
            # assert threading.active_count() == 1
            ...
            # fsspec may spawn a thread even if 'use_threads' is set to False (can this be avoided?)

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

        if hasattr(self, "_session") and self._session is not None:
            self._fs.close_session(self._executor.loop, self._session)
            self._session = None

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
        for start, stop in ranges:
            # _cat_file is async while cat_file is not.
            # Loop executor takes a coroutine while ThreadPoolExecutor takes a function.
            future = self._executor.submit(
                self._fs._cat_file if self._use_async else self._fs.cat_file,
                # it is assumed that the first argument is the file path / url (can have different names: 'url', 'path')
                self._file_path,
                start=start,
                end=stop,
            )
            chunk = uproot.source.chunk.Chunk(self, start, stop, future)
            future.add_done_callback(uproot.source.chunk.notifier(chunk, notifications))
            chunks.append(chunk)
        return chunks

    @property
    def use_async(self) -> bool:
        """
        True if using an async loop executor; False otherwise.
        """
        return self._use_async

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
        import fsspec.asyn

        return fsspec.asyn.get_loop()

    def submit(self, coroutine, /, *args, **kwargs) -> concurrent.futures.Future:
        loop = self.loop
        if not asyncio.iscoroutinefunction(coroutine):
            raise TypeError("loop executor can only submit coroutines")
        if not loop.is_running():
            raise RuntimeError("cannot submit coroutine while loop is not running")
        coroutine_object = coroutine(*args, **kwargs)
        return asyncio.run_coroutine_threadsafe(coroutine_object, loop)
