import uproot.source.chunk


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

    def __init__(self, file_path, **kwargs):
        import fsspec.core

        # Remove uproot-specific options (should be done earlier)
        # TODO: is timeout always valid?

        # TODO: import a list of all valid options instead of hardcoding
        exclude_keys = {
            "file_handler",
            "xrootd_handler",
            "http_handler",
            "s3_handler",
            "object_handler",
            "max_num_elements",
            "num_workers",
            "num_fallback_workers",
            "use_threads",
            "begin_chunk_size",
            "minimal_ttree_metadata",
        }

        opts = {k: v for k, v in kwargs.items() if k not in exclude_keys}

        self._fs, self._file_path = fsspec.core.url_to_fs(file_path, **opts)
        self._file = self._fs.open(self._file_path, "rb")
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

    def __enter__(self):
        self._fh = self._file.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._fh = None
        self._file.__exit__(exception_type, exception_value, traceback)

    def chunk(self, start, stop):
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

    def chunks(self, ranges, notifications):
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
        data = self._fs.cat_ranges(
            [self._file_path] * len(ranges),
            [start for start, _ in ranges],
            [stop for _, stop in ranges],
        )
        chunks = []
        for item, (start, stop) in zip(data, ranges):
            future = uproot.source.futures.TrivialFuture(item)
            chunk = uproot.source.chunk.Chunk(self, start, stop, future)
            uproot.source.chunk.notifier(chunk, notifications)()
            chunks.append(chunk)
        return chunks

    @property
    def num_bytes(self):
        """
        The number of bytes in the file.
        """
        return self._fs.size(self._file_path)

    @property
    def closed(self):
        """
        True if the associated file/connection/thread pool is closed; False
        otherwise.
        """
        return False
