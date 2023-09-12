# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This module defines a physical layer for file-like objects.

Defines a :doc:`uproot.source.object.ObjectResource` (wrapped Python file-like
object) and one source :doc:`uproot.source.object.ObjectSource` which always
has exactly one worker (we can't assume that the object is thread-safe).
"""


import uproot
import uproot.source.chunk
import uproot.source.futures


class ObjectResource(uproot.source.chunk.Resource):
    """
    Args:
        obj: The file-like object to use.

    A :doc:`uproot.source.chunk.Resource` for a file-like object.

    This object must have the following methods:

    - ``read(num_bytes)`` where ``num_bytes`` is an integer number of bytes to
      read.
    - ``seek(position)`` where ``position`` is an integer position to seek to.

    Both of these methods change the internal state of the object, its current
    seek position (because ``read`` moves that position forward ``num_bytes``).
    Hence, it is in principle not thread-safe.
    """

    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    @property
    def closed(self):
        return getattr(self._obj, "closed", False)

    def __enter__(self):
        if hasattr(self._obj, "__enter__"):
            self._obj.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if hasattr(self._obj, "__exit__"):
            self._obj.__exit__(exception_type, exception_value, traceback)

    def get(self, start, stop):
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a Python buffer of data between ``start`` and ``stop``.
        """
        self._obj.seek(start)
        return self._obj.read(stop - start)

    @staticmethod
    def future(source, start, stop):
        """
        Args:
            source (:doc:`uproot.source.object.ObjectSource`): The data source.
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a :doc:`uproot.source.futures.ResourceFuture` that calls
        :ref:`uproot.source.object.ObjectResource.get` with ``start`` and
        ``stop``.
        """

        def task(resource):
            return resource.get(start, stop)

        return uproot.source.futures.ResourceFuture(task)


class ObjectSource(uproot.source.chunk.MultithreadedSource):
    """
    Args:
        obj: The file-like object to use.

    A :doc:`uproot.source.chunk.Source` for a file-like object. (Although this
    is a :doc:`uproot.source.chunk.MultithreadedSource`, it never has more or
    less than one thread.)

    This object must have the following methods:

    - ``read(num_bytes)`` where ``num_bytes`` is an integer number of bytes to
      read.
    - ``seek(position)`` where ``position`` is an integer position to seek to.

    Both of these methods change the internal state of the object, its current
    seek position (because ``read`` moves that position forward ``num_bytes``).
    Hence, it is in principle not thread-safe.
    """

    ResourceClass = ObjectResource

    def __init__(self, obj, **options):
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = repr(obj)
        self._executor = uproot.source.futures.ResourceThreadPoolExecutor(
            [ObjectResource(obj)]
        )
        self._num_bytes = None


class BlockingObjectSource(uproot.source.chunk.Source):
    """
    Args:
        obj: The file-like object to use.

    A :doc:`uproot.source.chunk.Source` for a file-like object.

    This object must have the following methods:

    - ``read(num_bytes)`` where ``num_bytes`` is an integer number of bytes to
      read.
    - ``seek(position)`` where ``position`` is an integer position to seek to.

    Both of these methods change the internal state of the object, its current
    seek position (because ``read`` moves that position forward ``num_bytes``).

    Reading and seeking operations occur in the main thread.
    """

    _dtype = uproot.source.chunk.Chunk._dtype

    def __init__(self, obj, **options):
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._num_bytes = None
        self._file_path = repr(obj)
        self._obj = obj

    def __reduce__(self):
        return type(self), (self._obj,)

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        return f"<{type(self).__name__} {path} at 0x{id(self):012x}>"

    def chunk(self, start, stop):
        if self.closed:
            raise OSError(f"file {self._file_path} is closed")

        num_bytes = stop - start
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += num_bytes

        self._obj.seek(start)
        data = self._obj.read(num_bytes)

        future = uproot.source.futures.TrivialFuture(data)
        return uproot.source.chunk.Chunk(self, start, stop, future, is_memmap=True)

    def chunks(self, ranges, notifications):
        if self.closed:
            raise OSError(f"file {self._file_path} is closed")

        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        chunks = []
        for start, stop in ranges:
            num_bytes = stop - start
            self._obj.seek(start)
            data = self._obj.read(num_bytes)
            future = uproot.source.futures.TrivialFuture(data)
            chunk = uproot.source.chunk.Chunk(self, start, stop, future, is_memmap=True)
            notifications.put(chunk)
            chunks.append(chunk)
        return chunks

    @property
    def obj(self):
        """
        The underlying object
        """
        return self._obj

    @property
    def closed(self):
        return self._obj.closed  # TODO

    def __enter__(self):
        self._obj.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self._obj.__exit__(exception_type, exception_value, traceback)
