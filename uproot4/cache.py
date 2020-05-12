# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import threading

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

import uproot4._util


class LRUCache(MutableMapping):
    def __init__(self, limit_bytes):
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
        return self._limit_bytes

    @property
    def current_bytes(self):
        return self._current_bytes

    def __getitem__(self, where):
        return self._data[where]

    def __setitem__(self, where, what):
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
        with self._lock:
            self._current_bytes -= self._data[where]
            del self._data[where]
            self._order.remove(where)

    def __iter__(self):
        for x in self._order:
            yield x

    def __len__(self):
        return len(self._order)

    def get(self, where, function):
        try:
            return self._data[where]
        except KeyError:
            self._data[where] = function()
