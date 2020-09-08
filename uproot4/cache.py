# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Simple, thread-safe cache classes satisfying the ``MutableMapping`` protocol.

The :doc:`uproot4.cache.LRUCache` implements a least-recently used eviction
policy that limits the number of items in the cache (used as an
``object_cache``).

The :doc:`uproot4.cache.LRUArrayCache` implements the same policy, limiting the
total number of bytes, as reported by ``nbytes``.
"""

from __future__ import absolute_import

import threading

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

import uproot4._util


class LRUCache(MutableMapping):
    """
    Args:
        limit (None or int): Number of objects to allow in the cache before
            evicting the least-recently used. If None, this cache never evicts.

    LRUCache is a ``MutableMapping`` that evicts the least-recently used
    objects when the ``current`` number of objects exceeds the ``limit``.

    Get and set (or explicitly remove) items with ``MutableMapping`` syntax:
    square bracket subscripting.

    LRUCache is thread-safe for all options: getting, setting, deleting,
    iterating, listing keys, values, and items.

    This cache is insensitive to the size of the objects it stores, and hence
    is a better ``object_cache`` than an ``array_cache``.
    """

    @classmethod
    def sizeof(cls, obj):
        """
        The "size of" an object in this cache is always exactly 1.
        """
        return 1

    def __init__(self, limit):
        self._limit = limit
        self._current = 0
        self._order = []
        self._data = {}
        self._lock = threading.Lock()

    def __repr__(self):
        if self._limit is None:
            limit = "(no limit)"
        else:
            limit = "({0}/{1} full)".format(self._current, self._limit)
        return "<LRUCache {0} at 0x{1:012x}>".format(limit, id(self))

    @property
    def limit(self):
        """
        Number of objects to allow in the cache before evicting the
        least-recently used. If None, this cache never evicts.
        """
        return self._limit

    @property
    def current(self):
        """
        Current number of items in the cache.
        """
        return self._current

    def keys(self):
        """
        Returns a copy of the keys currently in the cache, in least-recently
        used order.

        The list ascends from least-recently used to most-recently used: index
        ``0`` is the least-recently used and index ``-1`` is the most-recently
        used.

        (Calling this method does not change the order.)
        """
        with self._lock:
            return list(self._order)

    def values(self):
        """
        Returns a copy of the values currently in the cache, in least-recently
        used order.

        The list ascends from least-recently used to most-recently used: index
        ``0`` is the least-recently used and index ``-1`` is the most-recently
        used.

        (Calling this method does not change the order.)
        """
        with self._lock:
            return [self._data[where] for where in self._order]

    def items(self):
        """
        Returns a copy of the items currently in the cache, in least-recently
        used order.

        The list ascends from least-recently used to most-recently used: index
        ``0`` is the least-recently used and index ``-1`` is the most-recently
        used.

        (Calling this method does not change the order.)
        """
        with self._lock:
            return [(where, self._data[where]) for where in self._order]

    def __getitem__(self, where):
        with self._lock:
            out = self._data[where]
            self._order.remove(where)
            self._order.append(where)
            return out

    def __setitem__(self, where, what):
        with self._lock:
            if where in self._data:
                self._order.remove(where)
            self._order.append(where)
            self._data[where] = what
            self._current += self.sizeof(what)

            if self._limit is not None:
                while self._current > self._limit and len(self._order) > 0:
                    key = self._order.pop(0)
                    self._current -= self.sizeof(self._data[key])
                    del self._data[key]

    def __delitem__(self, where):
        with self._lock:
            self._current -= self.sizeof(self._data[where])
            del self._data[where]
            self._order.remove(where)

    def __iter__(self):
        with self._lock:
            order = list(self._order)
        for x in order:
            yield x

    def __len__(self):
        with self._lock:
            return len(self._order)


class LRUArrayCache(LRUCache):
    """
    Args:
        limit_bytes (None, int, or str): Amount of data to allow in the cache
            before evicting the least-recently used. An integer is interpreted
            as a number of bytes and a string must be a number followed by a
            unit, such as "100 MB". If None, this cache never evicts.

    LRUArrayCache is a ``MutableMapping`` that evicts the least-recently used
    objects when the ``current`` number of bytes exceeds the ``limit``. The
    size of an object is taken from its ``nbytes`` attribute. If it does not
    have an ``nbytes``, it is presumed to have ``default_nbytes``.

    Get and set (or explicitly remove) items with ``MutableMapping`` syntax:
    square bracket subscripting.

    LRUArrayCache is thread-safe for all options: getting, setting, deleting,
    iterating, listing keys, values, and items.

    This cache is sensitive to the size of the objects it stores, but only if
    those objects have meaningful ``nbytes``. It is therefore a better
    ``array_cache`` than an ``array_cache``.
    """

    default_nbytes = 1024

    @classmethod
    def sizeof(cls, what):
        """
        The "size of" an object in this cache is its ``nbytes`` attribute or,
        if it doesn't have one, ``default_nbytes``.
        """
        return getattr(what, "nbytes", cls.default_nbytes)

    def __init__(self, limit_bytes):
        if limit_bytes is None:
            limit = None
        else:
            limit = uproot4._util.memory_size(limit_bytes)
        super(LRUArrayCache, self).__init__(limit)

    def __repr__(self):
        if self._limit is None:
            limit = "(no limit)"
        else:
            limit = "({0}/{1} bytes full)".format(self._current, self._limit)
        return "<LRUArrayCache {0} at 0x{1:012x}>".format(limit, id(self))

    @property
    def limit(self):
        """
        Number of bytes to allow in the cache before evicting the
        least-recently used. If None, this cache never evicts.
        """
        return self._limit

    @property
    def current(self):
        """
        Current number of bytes in the cache.
        """
        return self._current
