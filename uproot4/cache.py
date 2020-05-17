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
    Cache with Least-Recently Used (LRU) semantics, evicting if the total
    sum of all objects' `nbytes` exceeds `limit_bytes`.
    """

    def __init__(self, limit_bytes):
        """
        Args:
            limit_bytes (None, int, or str): If None, this cache never evicts;
                otherwise, the limit is interpreted as a memory_size.
        """
        if limit_bytes is None:
            self._limit_bytes = None
        else:
            self._limit_bytes = uproot4._util.memory_size(limit_bytes)
        self._current_bytes = 0
        self._order = []
        self._data = {}
        self._lock = threading.Lock()

    @property
    def limit_bytes(self):
        """
        Number of bytes before evicting or None if this cache never evicts.
        """
        return self._limit_bytes

    @property
    def current_bytes(self):
        """
        Current sum of `nbytes` of all objects in the cache.
        """
        return self._current_bytes

    def __getitem__(self, where):
        """
        Try to get an object from the cache. Raises `KeyError` if it is not
        found.

        (Thread-safe and lockless.)
        """
        return self._data[where]

    def __setitem__(self, where, what):
        """
        Adds an object to the cache and evicts if the new `current_bytes`
        exceeds `limit_bytes`.

        (Thread-safe with a lock.)
        """
        with self._lock:
            if where in self._data:
                self._order.remove(where)
            self._order.append(where)
            self._data[where] = what
            self._current_bytes += what.nbytes

            while (
                self._limit_bytes is not None
                and self._current_bytes > self._limit_bytes
            ):
                key = self._order.pop(0)
                self._current_bytes -= self._data[key]
                del self._data[key]

    def __delitem__(self, where):
        """
        Manually deletes an item from the cache.

        (Thread-safe with a lock.)
        """
        with self._lock:
            self._current_bytes -= self._data[where]
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

    def get(self, where, function):
        """
        Attempts to get an item. If the item is not available, `function` is
        called and its value is both added to the cache and returned.
        """
        try:
            return self._data[where]
        except KeyError:
            out = self._data[where] = function()
            return out
