# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Simple, thread-safe cache for arrays (objects with an `nbytes` property).
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
    Cache with Least-Recently Used (LRU) semantics, evicting if the `current`
    number of objects exceeds `limit`.
    """

    @classmethod
    def sizeof(cls, obj):
        return 1

    def __init__(self, limit):
        """
        Args:
            limit (None or int): If None, this cache never evicts;
                otherwise, objects are evicted when the number of
                items reachs the limit.
        """
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
        Limit before evicting or None if this cache never evicts.
        """
        return self._limit

    @property
    def current(self):
        """
        Current fill level of the cache; to be compared with `limit`.
        """
        return self._current

    def __getitem__(self, where):
        """
        Try to get an object from the cache. Raises `KeyError` if it is not
        found.
        """
        with self._lock:
            out = self._data[where]
            self._order.remove(where)
            self._order.append(where)
            return out

    def __setitem__(self, where, what):
        """
        Adds an object to the cache and evicts if the new `current`
        exceeds `limit`.
        """
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
        """
        Manually deletes an item from the cache.
        """
        with self._lock:
            self._current -= self.sizeof(self._data[where])
            del self._data[where]
            self._order.remove(where)

    def __iter__(self):
        """
        Iterates over the data in the cache.
        """
        for x in self._order:
            yield x

    def __len__(self):
        """
        Number of items in the cache.
        """
        return len(self._order)


class LRUArrayCache(LRUCache):
    """
    Cache with Least-Recently Used (LRU) semantics, evicting if the `current`
    sum of all objects' `nbytes` exceeds `limit`.

    If an object does not have an `nbytes` attribute, it is presumed to have
    `default_nbytes`.
    """

    default_nbytes = 1024

    @classmethod
    def sizeof(cls, what):
        return getattr(what, "nbytes", cls.default_nbytes)

    def __init__(self, limit_bytes):
        """
        Args:
            limit_bytes (None, int, or str): If None, this cache never evicts;
                otherwise, the limit is interpreted as a memory_size.
        """
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
