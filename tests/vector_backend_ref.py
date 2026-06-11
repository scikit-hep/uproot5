"""Importable backend factory for process workers (P0.2 capstone): vector behaviors travel by
IMPORT REF — never by pickling the behavior dict (it contains lambdas)."""

from __future__ import annotations

from typing import Any


def make_backend() -> Any:
    import vector
    from graphed_awkward import AwkwardBackend

    vector.register_awkward()
    return AwkwardBackend(behavior=vector.backends.awkward.behavior)
